import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif

def compute_correlation(x, corrtype):
    x1 = pd.DataFrame(x)
    coef_matrix = x1.corr(method= corrtype)
    return np.array(coef_matrix)

def rank_correlation(x, y):
    corrs = []
    for i in range(x.shape[1]):
        corr,_ = pearsonr(x[:,i], y)
        corrs.append(corr)
    rank = np.argsort(np.abs(corrs))[::-1]
    return rank

def rank_mutual(x, y):
    mutual_info = mutual_info_classif(x, y)
    rank = np.argsort(mutual_info)[::-1]
    return rank