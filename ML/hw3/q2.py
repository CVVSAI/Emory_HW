import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

def tune_dt(x, y, dparams, lsparams):

        dt = DecisionTreeClassifier()
        param_grid = {
            'max_depth': dparams,
            'min_samples_leaf': lsparams
        }

        grid_search = GridSearchCV(dt, param_grid, scoring='roc_auc', cv=5)
        grid_search.fit(x, y)
        best_depth = grid_search.best_params_['max_depth']
        best_leaf_samples = grid_search.best_params_['min_samples_leaf']
        best_auc = grid_search.best_score_
        result_dict = {
            "best-depth": best_depth,
            "best-leaf-samples": best_leaf_samples,
            "best-auc": best_auc,
        }

        return result_dict