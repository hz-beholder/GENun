
import joblib

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


class ShallowModel(object):
    def __init__(self):
        pass
    
    def scaler_data(self, data):
        return data
            
    def train(self, train_x, train_y, save_name=None):
        x_ = self.scaler_data(train_x)
        self.model.fit(x_, train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        x_ = self.scaler_data(test_x)
        post_ = self.model.predict_proba(x_)
        return post_

    def test_acc(self, test_x, test_y):
        x_ = self.scaler_data(test_x)
        pred_y = self.model.predict(x_)
        return accuracy_score(test_y, pred_y)

    def test_auc(self, test_x, test_y):
        x_ = self.scaler_data(test_x)
        pred_y = self.model.predict_proba(x_)
        return roc_auc_score(test_y, pred_y[:, 1])


class DT(ShallowModel):
    def __init__(self, max_leaf_nodes=10, random_state=0):
        super(DT, self).__init__()
        self.model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=random_state)

class RF(ShallowModel):
    def __init__(self, n_estimators=500, min_samples_leaf=30, random_state=0):
        super(RF, self).__init__()
        self.model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, 
                                            min_samples_leaf=min_samples_leaf)

class SVM(ShallowModel):
    def __init__(self, C=1.0, gamma='auto', kernel='rbf'):
        super(SVM, self).__init__()
        self.model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True)

    def scaler_data(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data

class MLP(ShallowModel):
    def __init__(self, lr=0.001, max_iter=400, hidden_layer_sizes=(20,)):  # (400, 200, 100)
        super(MLP, self).__init__()
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                        max_iter=max_iter, early_stopping=True, learning_rate_init=lr)

    def scaler_data(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data

class LR(ShallowModel):
    def __init__(self, solver='lbfgs', max_iter=600, multi_class='ovr', random_state=0, n_jobs=1):
        super(LR, self).__init__()
        self.model = LogisticRegression(random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, n_jobs=n_jobs)

    def scaler_data(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data
    
    def test_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(self.scaler_data(test_x))
        # return roc_auc_score(test_y, pred_y[:, 1])  # binary class classification AUC
        return roc_auc_score(test_y, pred_y[:, 1], multi_class="ovr", average=None)  # multi-class AUC



def get_model(model_name, **kwargs):
    if model_name == 'dt':
        max_leaf_nodes = kwargs.get('max_leaf_nodes', 10)
        random_state = kwargs.get('random_state', 0)
        return DT(max_leaf_nodes, random_state)
    elif model_name == 'lr':
        solver = kwargs.get('solver', 'lbfgs')
        max_iter = kwargs.get('max_iter', 600)
        multi_class = kwargs.get('multi_class', 'ovr')
        random_state = kwargs.get('random_state', 0)
        n_jobs = kwargs.get('n_jobs', 1)
        return LR(solver, max_iter, multi_class, random_state, n_jobs)
    elif model_name == 'svm':
        C = kwargs.get('C', 1.0)
        gamma = kwargs.get('gamma', 'auto')
        kernel = kwargs.get('kernel', 'rbf')
        return SVM(C, gamma, kernel)
    elif model_name == 'mlp':
        lr = kwargs.get('lr', 0.001)
        max_iter = kwargs.get('max_iter', 400)
        hidden_size = kwargs.get('hidden_layer_sizes', (20,))
        return MLP(lr=lr, max_iter=max_iter, hidden_layer_sizes=hidden_size)
    elif model_name == 'rf':
        n_estimators = kwargs.get('n_estimators', 500)
        min_samples_leaf = kwargs.get('min_samples_leaf', 30)
        random_state = kwargs.get('random_state', 0)
        return RF(n_estimators, min_samples_leaf, random_state)
    else:
        raise Exception("invalid model name")
    
    
