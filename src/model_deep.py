import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models
from transformers import ViTModel, ViTFeatureExtractor

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)
class AllCNN(nn.Module):
    def __init__(self, n_channels=3, num_classes=10, dropout=False, filters_percentage=1., batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )
        # self.

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

# pyright: reportMissingImports=true, reportUntypedBaseClass=false, reportGeneralTypeIssues=false



def get_model(model_name, feature_dims, num_classes, pretrained=True):
    # assert len(feature_dims) in [2, 3]
    
    in_dim = np.prod(feature_dims)
    channel_dim = feature_dims[-1] if len(feature_dims) == 3 else 1
    is_grayscale = len(feature_dims) == 2
    out_dim = num_classes
    
    if model_name == "mlps":
        return MLPNet5(in_dimensions=in_dim, num_classes=out_dim)
    elif model_name == "logistic":
        return LRTorchNet(in_dimensions=in_dim, num_classes=out_dim)
    elif model_name == "simple_cnn":
        return SimpleCNN(in_channels=channel_dim, num_classes=out_dim, img_size=feature_dims[0])
    elif model_name == "allcnn" or model_name == "ALLCNN":
        return AllCNN(num_classes=out_dim)
    elif model_name == "resnet18":
        if is_grayscale:
            if pretrained:
                raise Exception("resnet pretrained model is not available for grayscale images")
            else:
                return resnet18(num_classes, is_grayscale)
        else:
            if pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                model = models.resnet18()
            if num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, out_dim)
            return model
    elif model_name == "resnet34":
        if is_grayscale and pretrained:    
            raise Exception("resnet pretrained model is not available for grayscale images")
        if is_grayscale:
            return resnet34(num_classes, is_grayscale)
        else:
            if pretrained:
                model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                model = models.resnet34()
            if num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, out_dim)
            return model
    elif model_name == "resnet50":
        if is_grayscale and pretrained:    
            raise Exception("resnet pretrained model is not available for grayscale images")
        if is_grayscale:
            return resnet50(num_classes, is_grayscale)
        else:
            if pretrained:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                model = models.resnet50()
            if num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, out_dim)
            return model
    elif model_name == "densenet":
        if is_grayscale and pretrained:
            raise Exception("densenet pretrained model is not available for grayscale images")
        if is_grayscale:
            model = models.densenet121()
            model.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            if pretrained:
                model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            else:
                model = models.densenet121()
        if num_classes != model.classifier.out_features:
            model.classifier = nn.Linear(model.classifier.in_features, out_dim)
        return model
    elif model_name == "vgg":
        if is_grayscale and pretrained:
            raise Exception("vgg pretrained model is not available for grayscale images")
        if is_grayscale:
            model = models.vgg11_bn()
            model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        else:
            if pretrained:
                model = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)
            else:
                model = models.vgg11_bn()
        if num_classes != model.classifier[-1].out_features:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, out_dim)
        return model
    elif model_name == "vit":
        return ViT(num_classes)
        # model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
        model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        if num_classes != model.heads.head.out_features:
            model.head = nn.Linear(model.heads.head.in_features, out_dim)
            nn.init.xavier_uniform_(model.heads.head.weight)
            nn.init.constant_(model.heads.head.bias, 0)
        return model
    else:
        raise Exception("invalid net name")


class ViT(nn.Module):
    def __init__(self, num_classes=20):
        super(ViT, self).__init__()
        self.base = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(self.base.config.hidden_size, num_classes)   
        self.num_classes = num_classes
        self.relu = nn.ReLU()

    def forward(self, x):
        outputs = self.base(pixel_values=x)
        logits = self.fc(outputs.last_hidden_state[:,0])

        return logits
    

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes, img_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        # Determine the size after max-pooling
        reduced_size = img_size // 4  # Two 2x2 max-pooling operations
        self.fc1 = nn.Linear(64 * reduced_size * reduced_size, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc(x)
        return x

class LRTorchNet(nn.Module):
    def __init__(self, in_dimensions, num_classes=10):
        super(LRTorchNet, self).__init__()
        self.linear = nn.Linear(in_dimensions, num_classes)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class MLPNet5(nn.Module):
    def __init__(self, in_dimensions=168, num_classes=10):
        super(MLPNet5, self).__init__()
        self.fc1 = nn.Linear(in_dimensions, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc(x)
        # temperature = 4
        # x /= temperature
        # return F.log_softmax(x, dim=1)
        return x

class Affine(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x * self.weight + self.bias
    
class StandardLinearLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, beta=np.sqrt(0.1), w_sig = np.sqrt(2.0)):
        self.beta = beta
        self.w_sig = w_sig
        super(StandardLinearLayer, self).__init__(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/np.sqrt(self.in_features))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=self.beta)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, beta={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.beta)

class MLP(nn.Module):
    def __init__(self, num_layer=1, num_classes=10,  hidden_size=32, input_size=1024):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.layers = self._make_layers()

    def _make_layers(self):
        layer = []
        layer += [
            StandardLinearLayer(self.input_size, self.hidden_size), nn.ReLU()]
        for i in range(self.num_layer - 2):
            layer += [StandardLinearLayer(self.hidden_size, self.hidden_size)]
            layer += [nn.ReLU()]
        layer += [StandardLinearLayer(self.hidden_size, self.num_classes)]
        return nn.Sequential(*layer)

    def forward(self, x):
        x = x.reshape(x.size(0), self.input_size)
        return self.layers(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale=True):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        # probas = F.softmax(logits, dim=1)
        return logits  #, probas


def resnet18(num_classes, gray_scale=True):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                    layers=[2, 2, 2, 2],
                    num_classes=num_classes,
                    grayscale=gray_scale)
    return model

def resnet34(num_classes, gray_scale=True):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock, 
                    layers=[3, 4, 6, 3],
                    num_classes=num_classes,
                    grayscale=gray_scale)
    return model

def resnet50(num_classes, gray_scale=True):
    """Constructs a ResNet-50 model."""
    model = ResNet(block=Bottleneck, 
                    layers=[3, 4, 6, 3],
                    num_classes=num_classes,
                    grayscale=gray_scale)
    return model


####################################

def parameter_reinitilize(model):
    if isinstance(model, nn.Linear) or isinstance(model, nn.Conv2d):
        init.kaiming_normal_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0)
    if isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)

'''
Added by Shash: CIFAR Specific ResNets from https://github.com/akamaster/pytorch_resnet_cifar10
Adapted to the style of calls of Golatkar though.
'''
def _weights_init(model):
    init_mult = 0.01
    classname = model.__class__.__name__
    #print(classname)
    if isinstance(model, nn.Linear) or isinstance(model, nn.Conv2d):
        init.kaiming_normal_(model.weight)
        model.weight.requires_grad = False
        model.weight *= init_mult
        model.weight.requires_grad = True


def get_retrain_layers(model, name, ret):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.BatchNorm2d) or isinstance(model, nn.Linear):
        ret.append((model, name))
    for child_name, child in model.named_children():
        get_retrain_layers(child, f'{name}.{child_name}', ret)
    return ret

def reset_final_layers(model, num_retrain, logger, re_initialize=True, pre_retain=None):
    for param in model.parameters():
        param.requires_grad = False

    # pre_ret = None
    # if pretrain_path is not None:
    #     model_pre = copy.deepcopy(model)
    #     model_pre = load_pretrained(model_pre, modname, pretrain_path)
    #     pre_ret = getRetrainLayers(model_pre, 'M_pre', [])
    #     pre_ret.reverse()
    
    done = 0
    retain = get_retrain_layers(model, 'M', [])
    retain.reverse()
    for idx in range(len(retain)):
        if re_initialize:
            if isinstance(retain[idx][0], nn.Conv2d) or isinstance(retain[idx][0], nn.Linear):
                if pre_retain is not None:
                    retain[idx][0].weight, retain[idx][0].bias = pre_retain[idx][0].weight, pre_retain[idx][0].bias    
                    logger.info(f'Reinitialized layer: {retain[idx][1]}')
                else:
                    parameter_reinitilize(retain[idx][0])
                    logger.info(f'Reinitialized layer: {retain[idx][1]}')
        if isinstance(retain[idx][0], nn.Conv2d) or isinstance(retain[idx][0], nn.Linear):
            done += 1
        for param in retain[idx][0].parameters():
            param.requires_grad = True
        if done >= num_retrain:
            break

    return model
