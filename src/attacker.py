import pickle
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.spatial import distance

from defense import Defensor
from logger import create_logger
from extractor import ExtractBase

class Attackers(object):
    def __init__(self, logpath, logname):
        self.logger = create_logger(logpath, logname + '_' + self.__get_name__())
        self.attackers = None
    
    def __get_name__(self):
        return "Attackers"
        
    def set_attacker(self, attackers):
        self.attackers = attackers
    
    def reset(self):
        self.logger.info("***********  RESETTING ATTACKERS  ***********")
        self.attacker = None

    def train(self, attack_model, train_data, constructor, save_path=None):
        # self.logger.info("***********  LAUNCHING MODEL ATTACK  ***********")
        self.logger.info("******** Training Attack Models ********")

        self.attackers = list()
        label = train_data.label.astype('int')
        for _, aproc in enumerate(constructor._member_names_):
            # self.logger.info(f"Training the attack model for {aproc}")
            self.logger.info(f"\t ==> Traing attack model for {aproc}-based feature")
            
            atkmod_ = deepcopy(attack_model)
            ## train an attacker model for different constructed features
            atk_feat = np.asarray(train_data[aproc].tolist())                        
            atkmod_.train(atk_feat, label, save_path)
            test_acc = atkmod_.test_acc(atk_feat, label)
            self.logger.info(f"\t ==> Training ACC: {test_acc}")
            self.attackers.append(atkmod_)
        self.logger.info("Train attack models: Done!")
        return self.attackers
    
    def evaluate(self, data, constructor, info_str="Testing"):
        self.logger.info(f"******** {info_str} Attack Models ********")
        atk_data = pd.DataFrame(data=data["label"], columns=["label"])
        
        results = dict()
        label = data.label.astype('int')
        for idx, apoc in enumerate(constructor._member_names_):
            atker = self.attackers[idx]
            atk_feat = np.asarray(data[apoc].tolist())
            atk_data[apoc] = atker.predict_proba(atk_feat).tolist()
            ts_acc_ = atker.test_acc(atk_feat, label)
            ts_auc_ = atker.test_auc(atk_feat, label) if len(np.unique(label)) > 1 else 0
            results[apoc] = (ts_auc_, ts_acc_) if ts_auc_ > 0 else ts_acc_
            self.logger.info(f"\t ==> {info_str} attack model for {apoc} features: ACC: {ts_acc_}, AUC: {ts_auc_}")
        return atk_data, results

class AttackSource(object):
    def __init__(self):
        self._extractor = None
    
    @property
    def get_extractor(self):
        return self._extractor

    def set_extractor(self, extractor:ExtractBase):
        self._extractor = extractor
    
    def save_features(self, feature, save_path):
        with open(save_path, 'wb') as fout:
            pickle.dump({"feature": feature}, fout)

    def load_features(self, save_path):
        with open(save_path, 'rb') as fin:
            data = pickle.load(fin)
            
        return data["feature"]
    
    def build_source_data(self, forget_data, test_data, model, device='cuda'):        
        num_pos = len(forget_data)
        num_neg = len(test_data)
        feat_pos = self._extractor.extract_feature(model, forget_data, device)
        feat_neg = self._extractor.extract_feature(model, test_data, device)
        df_feat = pd.DataFrame(data={'original': np.vstack((feat_pos, feat_neg)).tolist(), 'label': [1]*num_pos + [0]*num_neg})
        return df_feat
    
    def data_reform(self, data, feature_constructor, is_defence=False, top_k=0):
        data_res = data.copy()
        if is_defence:
            # self.logger.info(f"==> constructing [Top-k (k = {top_k})] defense feature" )
            if  top_k == 0:
                data_res = Defensor.launch_label_defense(data_res)
            elif top_k != 0:
                data_res = Defensor.launch_topkpost_defense(data_res, top_k=top_k)
        
        # using different approaches from feature constructor to construct features.    
        for apoc in feature_constructor:
            feature_transform(apoc, data_res)

        return data_res

def feature_transform(method, df):
    from const import CONSTRUCTIONMETHOD
    assert method.name in CONSTRUCTIONMETHOD.__members__, "invalid feature construction method"

    df_ref = df
    name_ = str(method.name)
    df_ref[name_] = ""
    if method.value == CONSTRUCTIONMETHOD.RAW.value:
        df_ref[name_] = df_ref.original
    elif method.value == CONSTRUCTIONMETHOD.DIRECT_DIFFERENCE.value:
        df_ref[name_] = df.original - df.unlearn
    elif method.value == CONSTRUCTIONMETHOD.SORTED_DIFFERENCE.value:
        for idx, feat in enumerate(df.original):
            sort_idxs = np.argsort(feat[0, :])
            df.original[idx] = feat[0, sort_idxs].reshape((1, sort_idxs.size))
            df.unlearn[idx] = df.unlearn[idx][0, sort_idxs].reshape((1, sort_idxs.size))
        df_ref[name_] = df.original - df.unlearn
    elif method.value == CONSTRUCTIONMETHOD.L2_DISTANCE.value:
        for idx in range(df.shape[0]):
            euclidean = distance.euclidean(df.original[idx], df.unlearn[idx])
            df_ref[name_][idx] = np.full((1, 1), euclidean)
    elif method.value == CONSTRUCTIONMETHOD.DIRECT_CONCAT.value:
        for idx in range(df.shape[0]):
            concats = np.concatenate((df.original[idx], df.unlearn[idx]), axis=1)
            df_ref[name_][idx] = concats
    elif method.value == CONSTRUCTIONMETHOD.SORTED_CONCAT.value:
        for idx, feat in enumerate(df.original):
            sort_idxs = np.argsort(feat[0, :])
            org_ft = feat[0, sort_idxs].reshape((1, sort_idxs.size))
            unlearn_ft = df.unlearn[idx][0, sort_idxs].reshape((1, sort_idxs.size))
            concats = np.concatenate((org_ft, unlearn_ft), axis=1)
            df_ref[name_][idx] = concats
    else:
        raise Exception("invalid feature construction method")
    
    return df_ref

