import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CybSecEnsemble:
    def __init__(self, ae_weight=0.55, iso_weight=0.45):
        self.ae_weight = ae_weight
        self.iso_weight = iso_weight
        self.scaler_ae = MinMaxScaler()
        self.scaler_iso = MinMaxScaler()
        self.threshold = None

    def fit(self, mse_scores, iso_scores):
        scaled_mse = self.scaler_ae.fit_transform(np.array(mse_scores).reshape(-1, 1)).flatten()
        scaled_iso = self.scaler_iso.fit_transform((-np.array(iso_scores)).reshape(-1, 1)).flatten()
        ensemble_scores = (self.ae_weight * scaled_mse) + (self.iso_weight * scaled_iso)
        self.threshold = np.percentile(ensemble_scores, 95)
        
    def predict(self, mse_scores, iso_scores):
        scaled_mse = self.scaler_ae.transform(np.array(mse_scores).reshape(-1, 1)).flatten()
        scaled_iso = self.scaler_iso.transform((-np.array(iso_scores)).reshape(-1, 1)).flatten()
        ensemble_scores = (self.ae_weight * scaled_mse) + (self.iso_weight * scaled_iso)
        return ensemble_scores > self.threshold, ensemble_scores
