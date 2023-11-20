import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score
from sklearn.svm import SVC

def build_logr(train_x, test_x, train_y, test_y):


    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15, random_state=42)
    
    clf = LogisticRegression(penalty='none', solver='lbfgs')
    clf.fit(train_x, train_y)
    
    train_preds = clf.predict_proba(train_x)[:,1]
    val_preds = clf.predict_proba(val_x)[:,1] 
    test_preds = clf.predict_proba(test_x)[:,1]


    train_preds1 = clf.predict(train_x) 

    val_preds1 = clf.predict(val_x)

    test_preds1 = clf.predict(test_x)

    train_auc = roc_auc_score(train_y, train_preds)
    train_f1 = f1_score(train_y, train_preds1)
    train_f2 = fbeta_score(train_y, train_preds1, beta=2)

    val_auc = roc_auc_score(val_y, val_preds)
    val_f1 = f1_score(val_y, val_preds1)
    val_f2 = fbeta_score(val_y, val_preds1, beta=2)

    test_auc = roc_auc_score(test_y, test_preds)
    test_f1 = f1_score(test_y, test_preds1)
    test_f2 = fbeta_score(test_y, test_preds1, beta=2)
   
    
    results = {'train-auc': train_auc,
               'train-f1': train_f1,
               'train-f2': train_f2,
               'val-auc': val_auc,
               'val-f1': val_f1, 
               'val-f2': val_f2,
               'test-auc': test_auc,
               'test-f1': test_f1,
               'test-f2': test_f2,
               'params': {}}
    
    return results

def build_dt(train_x, test_x, train_y, test_y):

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15, random_state=42)

    params = {'max_depth': [2, 3, 5, 8, 10], 
            'min_samples_leaf': [3, 5, 10, 15, 20]}

    dt = DecisionTreeClassifier()


    grid = GridSearchCV(dt, param_grid=params, scoring='roc_auc', cv=5)
    grid.fit(train_x, train_y)


    best_dt = grid.best_estimator_
    best_dt.fit(train_x, train_y)


    train_preds = best_dt.predict_proba(train_x)[:,1]
    val_preds = best_dt.predict_proba(val_x)[:,1]
    test_preds = best_dt.predict_proba(test_x)[:,1]

    train_preds1 = best_dt.predict(train_x) 

    val_preds1 = best_dt.predict(val_x)

    test_preds1 = best_dt.predict(test_x)

    train_auc = roc_auc_score(train_y, train_preds)
    train_f1 = f1_score(train_y, train_preds1)
    train_f2 = fbeta_score(train_y, train_preds1, beta=2)

    val_auc = roc_auc_score(val_y, val_preds)
    val_f1 = f1_score(val_y, val_preds1)
    val_f2 = fbeta_score(val_y, val_preds1, beta=2)

    test_auc = roc_auc_score(test_y, test_preds)
    test_f1 = f1_score(test_y, test_preds1)
    test_f2 = fbeta_score(test_y, test_preds1, beta=2)


    results = {'train-auc': train_auc,
             'train-f1': train_f1,
             'train-f2': train_f2,
             'val-auc': val_auc,
             'val-f1': val_f1,
             'val-f2': val_f2,
             'test-auc': test_auc,
             'test-f1': test_f1,
             'test-f2': test_f2,
             'params': grid.best_params_}

    return results

def build_rf(train_x, test_x, train_y, test_y):

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15, random_state=42)
    
    params = {
    'n_estimators': [100],
    'max_depth': [2, 3, 5, 8, 10], 
    'min_samples_leaf': [3, 5, 10, 15, 20]}

    rf = RandomForestClassifier()

    grid = GridSearchCV(rf, param_grid=params, scoring='roc_auc', cv=5)
    grid.fit(train_x, train_y)

    best_rf = grid.best_estimator_
    best_rf.fit(train_x, train_y)
    
    train_preds = best_rf.predict_proba(train_x)[:,1]
    val_preds = best_rf.predict_proba(val_x)[:,1]
    test_preds = best_rf.predict_proba(test_x)[:,1]

    train_preds1 = best_rf.predict(train_x) 

    val_preds1 = best_rf.predict(val_x)

    test_preds1 = best_rf.predict(test_x)

    train_auc = roc_auc_score(train_y, train_preds)
    train_f1 = f1_score(train_y, train_preds1)
    train_f2 = fbeta_score(train_y, train_preds1, beta=2)

    val_auc = roc_auc_score(val_y, val_preds)
    val_f1 = f1_score(val_y, val_preds1)
    val_f2 = fbeta_score(val_y, val_preds1, beta=2)

    test_auc = roc_auc_score(test_y, test_preds)
    test_f1 = f1_score(test_y, test_preds1)
    test_f2 = fbeta_score(test_y, test_preds1, beta=2)


    results = {'train-auc': train_auc,
             'train-f1': train_f1,
             'train-f2': train_f2,
             'val-auc': val_auc,
             'val-f1': val_f1,
             'val-f2': val_f2,
             'test-auc': test_auc,
             'test-f1': test_f1,
             'test-f2': test_f2,
             'params': grid.best_params_}

    return results


def build_svm(train_x, test_x, train_y, test_y):

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15, random_state=42)
    
    params = {
    'kernel': ['linear', 'poly'],
    'degree': [2, 3, 4], 
    'C': [1]
    }

    svm = SVC(probability=True)

    grid = GridSearchCV(svm, param_grid=params, scoring='roc_auc', cv=5)
    grid.fit(train_x, train_y)

    best_svm = grid.best_estimator_
    best_svm.fit(train_x, train_y)
    
    train_preds = best_svm.predict_proba(train_x)[:,1]
    val_preds = best_svm.predict_proba(val_x)[:,1]
    test_preds = best_svm.predict_proba(test_x)[:,1]

    train_preds1 = best_svm.predict(train_x) 

    val_preds1 = best_svm.predict(val_x)

    test_preds1 = best_svm.predict(test_x)

    train_auc = roc_auc_score(train_y, train_preds)
    train_f1 = f1_score(train_y, train_preds1)
    train_f2 = fbeta_score(train_y, train_preds1, beta=2)

    val_auc = roc_auc_score(val_y, val_preds)
    val_f1 = f1_score(val_y, val_preds1)
    val_f2 = fbeta_score(val_y, val_preds1, beta=2)

    test_auc = roc_auc_score(test_y, test_preds)
    test_f1 = f1_score(test_y, test_preds1)
    test_f2 = fbeta_score(test_y, test_preds1, beta=2)


    results = {'train-auc': train_auc,
             'train-f1': train_f1,
             'train-f2': train_f2,
             'val-auc': val_auc,
             'val-f1': val_f1,
             'val-f2': val_f2,
             'test-auc': test_auc,
             'test-f1': test_f1,
             'test-f2': test_f2,
             'params': grid.best_params_}

    return results

def build_nn(train_x, test_x, train_y, test_y):

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15, random_state=42)

    param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (50,20), (100,), (100, 50)] ,
    'activation': ['logistic', 'tanh', 'relu'], 
    'alpha': [0.0001, 0.001, 0.01, 0.1]
    }
    
    nn = MLPClassifier()

    grid = GridSearchCV(nn, param_grid, cv=5, scoring='roc_auc')

    grid.fit(train_x, train_y)
    
    best_nn = grid.best_estimator_
    best_nn.fit(train_x, train_y)
    
    train_preds = best_nn.predict_proba(train_x)[:,1]
    val_preds = best_nn.predict_proba(val_x)[:,1]
    test_preds = best_nn.predict_proba(test_x)[:,1]

    train_preds1 = best_nn.predict(train_x) 

    val_preds1 = best_nn.predict(val_x)

    test_preds1 = best_nn.predict(test_x)

    train_auc = roc_auc_score(train_y, train_preds)
    train_f1 = f1_score(train_y, train_preds1)
    train_f2 = fbeta_score(train_y, train_preds1, beta=2)

    val_auc = roc_auc_score(val_y, val_preds)
    val_f1 = f1_score(val_y, val_preds1)
    val_f2 = fbeta_score(val_y, val_preds1, beta=2)

    test_auc = roc_auc_score(test_y, test_preds)
    test_f1 = f1_score(test_y, test_preds1)
    test_f2 = fbeta_score(test_y, test_preds1, beta=2)


    results = {'train-auc': train_auc,
             'train-f1': train_f1,
             'train-f2': train_f2,
             'val-auc': val_auc,
             'val-f1': val_f1,
             'val-f2': val_f2,
             'test-auc': test_auc,
             'test-f1': test_f1,
             'test-f2': test_f2,
             'params': grid.best_params_}

    return results
