import numpy as np
from scipy.optimize import fmin_ncg

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd.functional import vhp
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils import parameters_to_vector

from utils import make_functional, load_weights


def conjugate_gradient(ax_fn, b, debug_callback=None, avextol=None, maxiter=None):
    """Computes the solution to Ax - b = 0 by minimizing the conjugate objective
    f(x) = x^T A x / 2 - b^T x. This does not require evaluating the matrix A
    explicitly, only the matrix vector product Ax.

    From https://github.com/kohpangwei/group-influence-release/blob/master/influence/conjugate.py.

    Args:
      ax_fn: A function that return Ax given x.
      b: The vector b.
      debug_callback: An optional debugging function that reports the current optimization function. Takes two
          parameters: the current solution and a helper function that evaluates the quadratic and linear parts of the
          conjugate objective separately. (Default value = None)
      avextol:  (Default value = None)
      maxiter:  (Default value = None)

    Returns:
      The conjugate optimization solution.

    """

    cg_callback = None
    if debug_callback:
        cg_callback = lambda x: debug_callback(
            x, -np.dot(b, x), 0.5 * np.dot(x, ax_fn(x))
        )

    result = fmin_ncg(
        f=lambda x: 0.5 * np.dot(x, ax_fn(x)) - np.dot(b, x),
        x0=np.zeros_like(b),
        fprime=lambda x: ax_fn(x) - b,
        fhess_p=lambda x, p: ax_fn(p),
        callback=cg_callback,
        avextol=avextol,
        maxiter=maxiter,
    )

    return result


def s_test_cg(x_test, y_test, model, train_loader, damp, device="GPU", verbose=True):
    x_test, y_test = x_test.to(device), y_test.to(device)

    v_flat = parameters_to_vector(grad_z(x_test, y_test, model, gpu))

    def hvp_fn(x):
        x_tensor = torch.tensor(x, requires_grad=False)
        x_tensor = x_tensor.to(device)

        params, names = make_functional(model)
        # Make params regular Tensors instead of nn.Parameter
        params = tuple(p.detach().requires_grad_() for p in params)
        flat_params = parameters_to_vector(params)

        hvp = torch.zeros_like(flat_params)
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)

            def f(flat_params_):
                split_params = tensor_to_tuple(flat_params_, params)
                load_weights(model, names, split_params)
                out = model(x_train)
                loss = model.loss(out, y_train)
                return loss

            batch_hvp = vhp(f, flat_params, x_tensor, strict=True)[1]
            hvp += batch_hvp / float(len(train_loader))

        with torch.no_grad():
            load_weights(model, names, params, as_params=True)
            damped_hvp = hvp + damp * v_flat

        return damped_hvp.cpu().numpy()

    def print_function_value(_, f_linear, f_quadratic):
        print(
            f"Conjugate function value: {f_linear + f_quadratic}, lin: {f_linear}, quad: {f_quadratic}"
        )

    debug_callback = print_function_value if verbose else None

    result = conjugate_gradient(
        hvp_fn,
        v_flat.cpu().numpy(),
        debug_callback=debug_callback,
        avextol=1e-8,
        maxiter=100,
    )

    result = torch.tensor(result).to(device)
    return result


def calc_loss(y, t):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""
    ####################
    # if dim == [0, 1, 3] then dim=0; else dim=1
    ####################
    # y = F.log_softmax(y)
    y = F.log_softmax(y, dim=1)
    loss = F.nll_loss(y, t, weight=None, reduction='mean')
    return loss


def grad_z(z, t, model, device="cpu"):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: device for GPU or CPU
        
    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    # initialize
    model.eval()
    z, t = z.to(device), t.to(device)
    y = model(z)
    loss = calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    return list(grad(loss, params, create_graph=True))


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


def s_test(x_test, y_test, model, i, samples_loader, device="GPU", damp=0.01, scale=25.0):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        device: GPU or CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(x_test, y_test, model, device)
    h_estimate = v.copy()

    params, names = make_functional(model)
    req_grad = [p.requires_grad for p in params]
    # Make params regular Tensors instead of nn.Parameter and do not lose gradient information along the way
    params = tuple(p.detach().requires_grad_() if r else p.detach() for (r, p) in zip(req_grad, params))

    # Only calculate s_test with respect to parameters that require grad:
    params_grad = tuple(p for (r, p) in zip(req_grad, params) if r)
    names_grad = [n for (r, n) in zip(req_grad, names) if r]
    params_nograd = tuple(p for (r, p) in zip(req_grad, params) if not r)
    names_nograd = [n for (r, n) in zip(req_grad, names) if not r]

    # TODO: Dynamically set the recursion depth so that iterations stop once h_estimate stabilises
    # progress_bar = tqdm(samples_loader, desc=f"IHVP sample {i}")
    for i, (x_train, y_train) in enumerate(samples_loader): #enumerate(progress_bar):
        x_train, y_train = x_train.to(device), y_train.to(device)

        def f(*new_params):
            load_weights(model, names_grad, new_params)
            load_weights(model, names_nograd, params_nograd)
            out = model(x_train)
            loss = calc_loss(out, y_train)  #model.loss(out, y_train)
            return loss

        hv = vhp(f, params_grad, tuple(h_estimate), strict=True)[1]
        # Recursively calculate h_estimate
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]

            # if i % 100 == 0:
            #     norm = sum([h_.norm() for h_ in h_estimate])
                # progress_bar.set_postfix({"est_norm": norm.item()})

    with torch.no_grad():
        load_weights(model, names, params, as_params=True)
        for (r, p) in zip(req_grad, model.parameters()):
            p.requires_grad_(r)

    return h_estimate


def s_test_sample(model, x_test, y_test, train_loader, device="GPU", damp=0.01, scale=25.0, recursion_depth=5000, r=1):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        x_test: test image
        y_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        device: str, device for GPU or CPU
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""

    inverse_hvp = [
        torch.zeros_like(params, dtype=torch.float) for params in model.parameters() if params.requires_grad
    ]

    for i in range(r):
        hessian_loader = DataLoader(
            train_loader.dataset,
            sampler = RandomSampler(
                train_loader.dataset, True, num_samples=recursion_depth * train_loader.batch_size
            ),
            batch_size=train_loader.batch_size,
            num_workers=4,
        )

        cur_estimate = s_test(
            x_test, y_test, model, i, hessian_loader, device, damp=damp, scale=scale
        )

        with torch.no_grad():
            inverse_hvp = [
                old + cur for old, cur in zip(inverse_hvp, cur_estimate)
            ]  # (cur / scale)

    with torch.no_grad():
        inverse_hvp = [component / r for component in inverse_hvp]

    return inverse_hvp




def s_test_sim(x_test, y_test, model, train_loader, device="GPU", damp=0.01, scale=25.0, recursion_depth=5000):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(x_test, y_test, model, device)
    h_estimate = v.copy()

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        #########################
        for x, t in train_loader:
            x, t = x.to(device), t.to(device)
            y = model(x)
            loss = calc_loss(y, t)
            params = [ p for p in model.parameters() if p.requires_grad ]
            hv = hvp(loss, params, h_estimate)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
    return h_estimate

def calc_s_test_single(model, x_test, y_test, train_loader, device="GPU",
                       damp=0.01, scale=25, recursion_depth=5000, r=1):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. 
        s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        z_test: test image
        t_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""
    # s_test_vec_list = []
    # for i in range(r):
    #     s_test_vec_list.append(s_test_sim(x_test, y_test, model, train_loader,
    #                                   device=device, damp=damp, scale=scale,
    #                                   recursion_depth=recursion_depth))

    # ################################
    # # TODO: Understand why the first[0] tensor is the largest with 1675 tensor
    # #       entries while all subsequent ones only have 335 entries?
    # ################################
    # s_test_vec = s_test_vec_list[0]
    # for i in range(1, r):
    #     s_test_vec += s_test_vec_list[i]

    # s_test_vec = [i / r for i in s_test_vec]
    # return s_test_vec

    s_test_vec = None
    for _ in range(r):
        h_est = s_test_sim(x_test, y_test, model, train_loader, device=device, 
                           damp=damp, scale=scale, recursion_depth=recursion_depth)
        if s_test_vec is None:
            s_test_vec = h_est
        else:
            s_test_vec = [i + j for i, j in zip(s_test_vec, h_est)]
    
    s_test_vec = [i / r for i in s_test_vec]
    return s_test_vec
    

### calculate influence functions for samples

def calc_s_test(model, test_loader, train_loader, device="GPU",
                damp=0.01, scale=25, recursion_depth=5000, r=1, start=0):
    """Calculates s_test for the whole test dataset taking into account all
    training data images.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        test_loader: pytorch dataloader, which can load the test data
        train_loader: pytorch dataloader, which can load the train data
        device: string, device for GPU for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        start: int, index of the first test index to use. default is 0

    Returns:
        s_tests: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge."""

    s_tests = []
    for i in range(start, len(test_loader.dataset)):
        z_test, t_test = test_loader.dataset[i]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        s_test_vec = calc_s_test_single(model, z_test, t_test, train_loader,
                                        device, damp, scale, recursion_depth, r)
        s_tests.append(s_test_vec)
    return s_tests


def calc_grad_z(model, train_loader, device="cpu", start=0):
    """Calculates grad_z. One grad_z should be computed for each training data sample.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        train_loader: pytorch dataloader, which can load the train data
        device: int, device to use for GPU or CPU (default)
        start: int, index of the first test index to use. default is 0

    Returns:
        grad_zs: list of torch tensors, contains the grad_z tensors
    """

    grad_zs = []
    for i in range(start, len(train_loader.dataset)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        grad_z_vec = grad_z(z, t, model, device)
        grad_zs.append(grad_z_vec)

    return grad_zs



def calc_influence_function(train_dataset_size, grad_z_vecs=None, e_s_test=None):
    """Calculates the influence function

    Arguments:
        train_dataset_size: int, total train dataset size
        grad_z_vecs: list of torch tensor, containing the gradients
            from model parameters to loss
        e_s_test: list of torch tensor, contains s_test vectors

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness"""
    if len(grad_z_vecs) != train_dataset_size:
        train_dataset_size = len(grad_z_vecs)

    influences = []
    for i in range(train_dataset_size):
        tmp_influence = (
            sum(
                [
                    ###################################
                    # TODO: verify if computation really needs to be done
                    # on the CPU or if GPU would work, too
                    ###################################
                    torch.sum(k * j).data.cpu().numpy()
                    for k, j in zip(grad_z_vecs[i], e_s_test)
                    ###################################
                    # Originally with [i] because each grad_z contained
                    # a list of tensors as long as e_s_test list
                    # There is one grad_z per training data sample
                    ###################################
                ]
            )
            / train_dataset_size
        )
        influences.append(tmp_influence.cpu())
        # display_progress("Calc. influence function: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist()
