from copy import deepcopy
import numpy as np

class Defensor(object):
    @staticmethod
    def launch_topkpost_defense(posterior_df, top_k):
        posterior_df = deepcopy(posterior_df)

        for index, posterior in enumerate(posterior_df.original):
            sort_indices = np.argsort(posterior[0, :])
            sum = 0
            for k in range(top_k):
                sum += posterior[0, sort_indices[-(k+1)]]
            ave = (1-sum) / (sort_indices.size - top_k)

            for i in range(sort_indices.size):
                if i in range(top_k):
                    posterior_df.original[index][0][sort_indices[-(i+1)]] = posterior[0, sort_indices[-(i+1)]]
                else:
                    posterior_df.original[index][0][sort_indices[-(i+1)]] = ave

        for index, posterior in enumerate(posterior_df.unlearning):
            sort_indices = np.argsort(posterior[0, :])
            sum = 0
            for k in range(top_k):
                sum += posterior[0, sort_indices[-(k + 1)]]
            ave = (1 - sum) / (sort_indices.size - top_k)

            for i in range(sort_indices.size):
                if i in range(top_k):
                    posterior_df.unlearning[index][0][sort_indices[-(i+1)]] = posterior[0, sort_indices[-(i+1)]]
                else:
                    posterior_df.unlearning[index][0][sort_indices[-(i+1)]] = ave
        return posterior_df

    @staticmethod
    def launch_label_defense(posterior_df):
        posterior_df = deepcopy(posterior_df)
        
        for index, posterior in enumerate(posterior_df.original):
            sort_indices = np.argsort(posterior[0, :])
            for i in range(sort_indices.size):
                if i == sort_indices[-1]:
                    posterior_df.original[index][0][i] = 1
                else:
                    posterior_df.original[index][0][i] = 0
        
        for index, posterior in enumerate(posterior_df.unlearning):
            sort_indices = np.argsort(posterior[0, :])
            for i in range(sort_indices.size):
                if i == sort_indices[-1]:
                    posterior_df.unlearning[index][0][i] = 1
                else:
                    posterior_df.unlearning[index][0][i] = 0

        return posterior_df
