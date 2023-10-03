import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

def generate_train_val(x, y, val_size):
    shuffle_data = np.random.permutation(len(x))
    x = np.array(x)[shuffle_data]
    y = np.array(y)[shuffle_data]
    val = int(len(x) * val_size)
    x_train, x_val = x[:-val], x[-val:]
    y_train, y_val = y[:-val], y[-val:]
    results = {
    'train-x': x_train,
    'train-y': y_train,
    'val-x': x_val, 
    'val-y': y_val
    }
    return results

def generate_kfold(x, y, k):
    indices = np.arange(len(x))
    fold_size = len(x) // k
    np.random.shuffle(indices)
    folds = np.zeros(len(x))
    for i in range(k):
        start = i * fold_size
        end = (i+1) * fold_size
        fold_indices = indices[start:end]
        folds[fold_indices] = i
    remainder = len(x) % k
    remainder_indices = indices[-remainder:]
    np.random.shuffle(remainder_indices)
    for i, ids in enumerate(remainder_indices):
        folds[ids] = i % k
    return folds

def eval_holdout(x, y, val_size, logistic):
    
    data = generate_train_val(x, y, val_size)
    x_train, x_val = data['train-x'], data['val-x'] 
    y_train, y_val = data['train-y'], data['val-y']
    logistic.fit(x_train, y_train)
    
    train_pred = logistic.predict(x_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, logistic.predict_proba(x_train)[:,1])

    val_pred = logistic.predict(x_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, logistic.predict_proba(x_val)[:,1])
    
    results = {
    'train-acc': train_acc, 
    'train-auc': train_auc,
    'val-acc': val_acc,
    'val-auc': val_auc
    }
    
    return results

def eval_kfold(x, y, k, logistic):
    folds = generate_kfold(x, y, k)
    train_accuracies = []
    train_aucs = []
    val_accuracies= []
    val_aucs = []
    for fold in range(k):
        train_indices = folds != fold
        val_indices = folds == fold
        x_train, x_val = x[train_indices], x[val_indices] 
        y_train, y_val = y[train_indices], y[val_indices]
        logistic.fit(x_train, y_train)
        train_accuracies.append(accuracy_score(y_train, logistic.predict(x_train)))
        train_aucs.append(roc_auc_score(y_train, logistic.predict_proba(x_train)[:,1]))
        val_accuracies.append(accuracy_score(y_val, logistic.predict(x_val))) 
        val_aucs.append(roc_auc_score(y_val, logistic.predict_proba(x_val)[:,1]))
    results = {
            'train-acc': np.mean(train_accuracies), 
            'train-auc': np.mean(train_aucs),
            'val-acc' : np.mean(val_accuracies),
            'val-auc' : np.mean(val_aucs)
            }
    return results

def eval_mccv(x, y, val_size, s, logistic):
    train_accuracies = []
    train_aucs = []
    val_accuracies= []
    val_aucs = []
    for i in range(s):
        A = eval_holdout(x, y, val_size, logistic)
        train_accuracies.append(A['train-acc'])
        train_aucs.append(A['train-auc'])
        val_accuracies.append(A['val-acc'])
        val_aucs.append(A['val-auc'])
    results = {
            'train-acc': np.mean(train_accuracies), 
            'train-auc': np.mean(train_aucs),
            'val-acc' : np.mean(val_accuracies),
            'val-auc' : np.mean(val_aucs)
            }
    return results
