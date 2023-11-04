from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

def tune_nn(x, y, hidden_params, act_params, alpha_params):
    best_auc = 0
    best_config = {}
    
    for hidden in hidden_params:
        for activation in act_params:
            for alpha in alpha_params:
                clf = MLPClassifier(hidden_layer_sizes=hidden, activation=activation, alpha=alpha)
                clf.fit(x, y)
                pred = clf.predict_proba(x)[:,1]
                auc = roc_auc_score(y, pred)
                
                if auc > best_auc:
                    best_auc = auc
                    best_config['best-hidden'] = hidden
                    best_config['best-activation'] = activation
                    best_config['best-alpha'] = alpha
    
    return best_config