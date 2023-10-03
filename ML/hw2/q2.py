import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

def do_nothing(train, test) :
    return train, test

def do_std(train, test): 
    scaler = StandardScaler() 
    scaler.fit(train)
    train_std = scaler.transform(train)
    test_std = scaler.transform(test)
    return train_std, test_std

def do_log(train, test):
    train = np.log(train + 0.1)
    test = np.log(test + 0.1)
    return train, test

def do_bin(train, test): 
    train = np.where(train > 0, 1, 0)
    test = np.where(test > 0, 1, 0)
    return train, test

def eval_nb(trainx, trainy, testx, testy):
    model = GaussianNB()
    model.fit(trainx, trainy)
    train_pred = model.predict(trainx)
    test_pred = model.predict(testx)
    test_prob = model.predict_proba(testx)[:, 1]
    results = {
    'train-acc': accuracy_score(trainy, train_pred),
    'train-auc': roc_auc_score(trainy, model.predict_proba(trainx)[:, 1]),
    'test-acc': accuracy_score(testy, test_pred), 
    'test-auc': roc_auc_score(testy, test_prob),
    'test-prob': test_prob
    }
    return results

def eval_lr(trainx, trainy, testx, testy):
    model = LogisticRegression()
    model.fit(trainx, trainy)
    train_pred = model.predict(trainx)
    test_pred = model.predict(testx)
    test_prob = model.predict_proba(testx)[:,1]
    results = {
    'train-acc': accuracy_score(trainy, train_pred), 
    'train-auc': roc_auc_score(trainy, model.predict_proba(trainx)[:,1]),
    'test-acc': accuracy_score(testy, test_pred),
    'test-auc': roc_auc_score(testy, test_prob),
    'test-prob': test_prob
    }
    return results

