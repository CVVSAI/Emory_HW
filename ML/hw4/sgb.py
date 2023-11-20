from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

def tune_sgtb(x, y, lst_nIter, lst_nu, lst_q, md):

    param_grid = {'nIter': lst_nIter,
                'nu': lst_nu,
                'q': lst_q}

    neg_mse = make_scorer(mean_squared_error, greater_is_better=False)
    gs = GridSearchCV(SGTB(), param_grid, cv=5, scoring=neg_mse)
    gs.fit(x, y)
    best_params = gs.best_params_

    best_nIter = best_params['nIter'] 
    best_nu = best_params['nu']
    best_q = best_params['q']

    return {'best-nIter': best_nIter, 
          'best-nu': best_nu,
          'best-q': best_q}

def compute_residual(y_true, y_pred):
    return  y_true - y_pred

class SGTB(RegressorMixin):
    def __init__(self, nIter=1, q=1, nu=0.1, md=3):
        self.nIter = nIter
        self.q = q
        self.nu = nu
        self.md = md
        self.y_mean = 0
        self.train_dict = {}
        self.trees = []

    def fit(self, x, y):
        self.y_mean = np.mean(y)
        y_pred = np.repeat(self.y_mean, len(y))
        r = compute_residual(y.reshape(len(y),), y_pred)
        self.train_dict[0] = np.sqrt(np.mean(r**2))
        for i in range(1, self.nIter+1):
            subsample_idx = np.random.choice(len(x), size=int(len(x)*self.q), replace=False)
            x_sub = x[subsample_idx] 
            r_sub = r[subsample_idx]
            tree = DecisionTreeRegressor(max_depth=self.md)
            tree.fit(x_sub, r_sub)
            self.trees.append(tree)
            y_pred = y_pred + self.nu * tree.predict(x)
            r = compute_residual(y.reshape(len(y),), y_pred)
            self.train_dict[i] = np.sqrt(np.mean(r**2))
        return self
    
    def get_params(self, deep=True):
        return {"nIter": self.nIter,
                "q": self.q,
                "nu": self.nu,
                "md": self.md}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, x):
        y_pred = np.repeat((self.y_mean), len(x))
        for i in range(self.nIter):
            tree = self.trees[i]
            update = self.nu * tree.predict(x)
            y_pred = y_pred + update
        return y_pred