import numpy as np

class SampleSelector:
    def __init__(self, model):
        self.model = model

    def select_high_entropy_samples(self, data, threshold):
        predictions = self.model.predict(data)
        entropy = -np.sum(predictions * np.log(predictions), axis=1)
        high_entropy_samples = data[entropy > threshold]
        return high_entropy_samples

    def select_outliers(self, data, threshold):
        predictions = self.model.predict(data)
        loss = np.max(predictions, axis=1)
        outliers = data[loss > threshold]
        return outliers

    def select_top_k_highest_loss_samples(self, data, topk):
        predictions = self.model.predict(data)
        loss = np.max(predictions, axis=1)
        top_k_samples = data[np.argsort(loss)[-topk:]]
        return top_k_samples

    def select_samples_near_decision_boundary(self, data, threshold):
        predictions = self.model.predict(data)
        max_prob = np.max(predictions, axis=1)
        near_boundary_samples = data[max_prob > threshold]
        return near_boundary_samples
        