from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


def tune_nn(x, y, hidden_params, act_params, alpha_params):

    model = MLPClassifier()

    param_grid = {
    'hidden_layer_sizes': hidden_params,
    'activation': act_params, 
    'alpha': alpha_params
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
    grid.fit(x, y)

    best_params = {
    'best-hidden': grid.best_params_['hidden_layer_sizes'],
    'best-activation': grid.best_params_['activation'],
    'best-alpha': grid.best_params_['alpha']
    }

    return best_params