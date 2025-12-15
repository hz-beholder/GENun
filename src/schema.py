from abc import abstractmethod
from torch.utils.data import DataLoader

# from defense import Defensor
# from attacker import AttackModel
# from const import DATATYPE
from logger import create_logger
from model_bases import ShallowModels, DeepModels


class BasicSchema(object):
    def __init__(self, logpath, logname):
        self.logpath = logpath
        self.logname = logname
    
    def set_basic(self, feature_dimension, num_classes):
        self.feature_dimension = feature_dimension
        self.num_classes = num_classes

    def get_model(self, model_name, feature_dimension, num_classes, pretrained=True):
        from model_bases import ShallowModels, DeepModels
        if ShallowModels.is_valid(model_name):
            return ShallowModels(model_name, self.logpath, self.logname)
        elif DeepModels.is_valid(self.model_name):
            return DeepModels(model_name, feature_dimension, num_classes, self.logpath, self.logname, pretrained)
        else:
            raise Exception("Invalid original model")


class BasicUnlearnSchema(BasicSchema):
    def __init__(self, logpath, logname) -> None:
        super().__init__(logpath, logname)
        self.logger = create_logger(self.logpath, '_'.join([self.logname, self.__get_name__()]))
    
    def __get_name__(self) -> str:
        return "UnlearnSchema"
    
    @abstractmethod
    def model_unlearn(self, args, data, batch_size, device, ckpt_path, **kwargs):
        pass

    def _train_(self, model:DeepModels, train_data, valid_data, test_data, ckpt_path, 
                batch_size=128, device="cuda", mask=None, num_workers=128):
        self.logger.info('Training start ....')
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # If GA-KL regularization is requested, build a retain loader and attach to model.params
        try:
            if hasattr(model, 'params') and model.params is not None and model.params.get('use_ga_kl', False):
                retain_dataset = model.params.get('retain_dataset', None)
                if retain_dataset is not None:
                    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    model.params['retain_loader'] = retain_loader
        except Exception:
            # keep backward compatibility on any failure
            pass
        best_raw_model = model.train_(train_loader, test_loader, valid_loader, ckpt_path, device, mask)
        self.logger.info('Train done!')
        return model
    
    @staticmethod
    def _test_model_(model:DeepModels, data, batch_size=256, lossfn='ce', device="cuda"):
        if data is None or len(data) <= 0:
            return 0.0, 0.0
        
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=64)
        acc_ = model.test_acc(loader, lossfn, device=device)
        try:
            auc_ = model.test_auc(loader, device=device)
        except:
            auc_ = 0.0
        return acc_, auc_


# class BasicAttackSchema(BasicSchema):
#     def __init__(self, logpath, logname):
#         super(BasicAttackSchema, self).__init__(logpath, logname)
#         self.attackers = None
#         self.logger = create_logger(self.logpath, '_'.join([self.logname, self.__get_name__()]))
    
#     def __get_name__(self) -> str:
#         return "AttackSchema"
    
#     def get_attacker(self):
#         return self.attackers
    
#     def set_attackers(self, attackers):
#         self.attackers = attackers
    
#     def reset(self):
#         self.attackers = None
    
#     @abstractmethod
#     def _obtain_features(self, num_models, original_model_path, target_model_path):
#         pass
    
#     def _save_feature_(self, feature_data, index, save_path):
#         with open(save_path, 'wb') as f:
#             pickle.dump({"feature": feature_data, "index": index}, f)

#     def _load_feature_(self, save_path):
#         with open(save_path, 'rb') as f:
#             data = pickle.load(f)
#         return data["index"], data["feature"]

#     def train_attackers(self, model_name, in_feature, save_path, feature_name='original'):
#         self.logger.info("******** Training attack models ********")

#         label = in_feature.label.astype('int')
#         # for _, attr in enumerate(self.construct_apoc._member_names_):            
#         self.logger.info(f"\t ==> attack model ({model_name}) for raw feature")

#         atk_feat = np.asarray(in_feature[feature_name].tolist()) #self._select_feature_(in_feature, attr)
#         dims = atk_feat.shape[1]
#         atk_model = AttackModel(self.args, model_name, dims, self.logger)
#         self.logger.info(f"Training the attack model")
#         atk_model.train(atk_feat, label, save_path)
#         self.attackers = atk_model
            
#         self.logger.info("Train attack models: Done!")
#         return self.attackers
    
#     def evaluation(self, data, feature_name='original'):
#         self.logger.info("Testing the attack model")
#         atk_data = pd.DataFrame(data=data["label"], columns=["label"])
        
#         results = dict()
#         label = data.label.astype('int')
#         atk_feat = np.asarray(data[feature_name].tolist())  #self._select_feature_(data, apoc)
#         atk_data[feature_name] = self.attackers.predict_proba(atk_feat).tolist()
#         acc, auc = self.attackers.test(atk_feat, label)
#         results[feature_name] = (acc, auc) if auc > 0 else acc
#         return atk_data, results
    
        
#     def select_samples(self, data, indices, dataset_name="", dataset_type:DATATYPE=DATATYPE.IMAGE):
#         # dataset_type = DataStore.dataset_type(self.dataset)
#         if dataset_type == DATATYPE.IMAGE:
#             data_ = deepcopy(self.data)
#             data_.data, data_.targets = data.data[indices], data.targets[indices]
#             return data_
#         elif dataset_type == DATATYPE.CATEGORY:
#             xs = data.iloc[indices, :-1]
#             ys = data.iloc[indices, -1]
#             if dataset_name == 'location':
#                 xs = np.concatenate((xs, np.zeros((9, xs.shape[1]))))
#             return (xs, ys)
#             # return self.df.values[index, :self.feature_dims].reshape([1, self.feature_dims])            
#         else:
#             raise Exception("invalid test dataset")

    
    # def construct_feature(self, df):
    #     self.logger.info("==> constructing feature")
    #     if self.args.is_defence:
    #         self.logger.info(f"==> constructing [Top-k (k = {self.args.top_k})] defence feature" )
    #         if  self.args.top_k == 0:
    #             df = Defensor.launch_label_defense(df)
    #         elif self.args.top_k != 0:
    #             df = Defensor.launch_topkpost_defense(df, top_k=self.args.top_k)
        
    #     # using different approaches to construct features.    
    #     for apoc in self.construct_apoc:
    #         self._obtain_feature_(apoc, df)
    
    # def _obtain_feature_(self, method, df):
    #     assert method.name in CONSTRUCTIONMETHOD.__members__, "invalid feature construction method"

    #     feature = df
    #     name_ = str(method.name)
    #     feature[name_] = ""

    #     if method.value == CONSTRUCTIONMETHOD.RAW.value:
    #         feature[name_] = feature.original
    #     elif method.value == CONSTRUCTIONMETHOD.DIRECT_DIFFERENCE.value:
    #         feature[name_] = df.original - df.unlearn
    #     elif method.value == CONSTRUCTIONMETHOD.SORTED_DIFFERENCE.value:
    #         for index, feat in enumerate(df.original):
    #             sort_idxs = np.argsort(feat[0, :])
    #             df.original[index] = feat[0, sort_idxs].reshape((1, sort_idxs.size))
    #             df.unlearn[index] = df.unlearn[index][0, sort_idxs].reshape((1, sort_idxs.size))
    #         feature[name_] = df.original - df.unlearn
    #     elif method.value == CONSTRUCTIONMETHOD.L2_DISTANCE.value:
    #         for index in range(df.shape[0]):
    #             euclidean = distance.euclidean(df.original[index], df.unlearn[index])
    #             feature[name_][index] = np.full((1, 1), euclidean)
    #     elif method.value == CONSTRUCTIONMETHOD.DIRECT_CONCAT.value:
    #         for index in range(df.shape[0]):
    #             concats = np.concatenate((df.original[index], df.unlearn[index]), axis=1)
    #             feature[name_][index] = concats
    #     elif method.value == CONSTRUCTIONMETHOD.SORTED_CONCAT.value:
    #         for index, feat in enumerate(df.original):
    #             sort_idxs = np.argsort(feat[0, :])
    #             org_ft = feat[0, sort_idxs].reshape((1, sort_idxs.size))
    #             unlearn_ft = df.unlearn[index][0, sort_idxs].reshape((1, sort_idxs.size))
    #             concats = np.concatenate((org_ft, unlearn_ft), axis=1)
    #             feature[name_][index] = concats
    #     else:
    #         raise Exception("invalid feature construction method")
        
    #     return feature
