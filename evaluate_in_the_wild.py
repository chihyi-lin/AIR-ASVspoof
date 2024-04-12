from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve

"""The calculate_eer function adapt from https://github.com/piotrkawa/deepfake-whisper-features/blob/main/src/metrics.py"""


def calculate_eer(y, y_score) -> Tuple[float, float, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y, -y_score)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    return thresh, eer, fpr, tpr

def compute_eer_in_the_wild(score_file):
    with open(score_file, 'r') as f:
        lines = f.readlines()
        y_true = []
        y_pred = []
        for line in lines:
            part = line.strip().split()
            y_true.append(part[1])
            y_pred.append(float(part[2]))
        inverted_y_true = 1 - np.array([1 if label == "spoof" else 0 for label in y_true])
        y_pred = np.array(y_pred)
    return calculate_eer(inverted_y_true, y_pred)

if __name__ == "__main__": 
    thresh, eer, fpr, tpr = compute_eer_in_the_wild('models_asv21_mfcc_eer_1004/ocsoftmax/in_the_wild_score_1.txt')
    print(f'EER In-the-wild: {eer:.4f}, thresh: {-thresh}')