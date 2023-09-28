import numpy as np
import sklearn as sk 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
from numpy import sqrt

def preprocess_data(trainx, valx, testx):
    train_mean = np.mean(trainx, axis=0)
    train_std = np.std(trainx, axis=0)
    trainx = (trainx - train_mean) / train_std
    valx = (valx - train_mean) / train_std
    testx = (testx - train_mean) / train_std
    return trainx, valx, testx


def eval_linear1(trainx, trainy, valx, valy, testx, testy):
    model = LinearRegression()
    data = {}
    model.fit(trainx, trainy)
    data["train-rmse"] =sqrt(mean_squared_error(trainy, model.predict(trainx)))
    data["train-r2"] = metrics.r2_score(trainy, model.predict(trainx))
    data["val-rmse"] =sqrt(mean_squared_error(valy, model.predict(valx)))
    data["val-r2"] = metrics.r2_score(valy, model.predict(valx))
    data["test-rmse"] =sqrt(mean_squared_error(testy, model.predict(testx)))
    data["test-r2"] = metrics.r2_score(testy, model.predict(testx))
    return data
    


def eval_linear2(trainx, trainy, valx, valy, testx, testy):
     comb_datax = np.concatenate((trainx, valx))
     comb_datay = np.concatenate((trainy, valy))
     model = LinearRegression()
     data = {}
     model.fit(comb_datax, comb_datay)
     data["train-rmse"] =sqrt(mean_squared_error(trainy, model.predict(trainx)))
     data["train-r2"] = metrics.r2_score(trainy, model.predict(trainx))
     data["val-rmse"] =sqrt(mean_squared_error(valy, model.predict(valx)))
     data["val-r2"] = metrics.r2_score(valy, model.predict(valx))
     data["test-rmse"] =sqrt(mean_squared_error(testy, model.predict(testx)))
     data["test-r2"] = metrics.r2_score(testy, model.predict(testx))
     return data

def eval_ridge1(trainx, trainy, valx, valy, testx, testy, alpha):
     model = Ridge(alpha = alpha)
     data = {}
     model.fit(trainx,trainy)
     data["train-rmse"] =sqrt(mean_squared_error(trainy, model.predict(trainx)))
     data["train-r2"] = metrics.r2_score(trainy, model.predict(trainx))
     data["val-rmse"] =sqrt(mean_squared_error(valy, model.predict(valx)))
     data["val-r2"] = metrics.r2_score(valy, model.predict(valx))
     data["test-rmse"] =sqrt(mean_squared_error(testy, model.predict(testx)))
     data["test-r2"] = metrics.r2_score(testy, model.predict(testx))
     return data



def eval_lasso1(trainx, trainy, valx, valy, testx, testy, alpha):
     model = Lasso(alpha = alpha)
     data = {}
     model.fit(trainx,trainy)
     data["train-rmse"] =sqrt(mean_squared_error(trainy, model.predict(trainx)))
     data["train-r2"] = metrics.r2_score(trainy, model.predict(trainx))
     data["val-rmse"] =sqrt(mean_squared_error(valy, model.predict(valx)))
     data["val-r2"] = metrics.r2_score(valy, model.predict(valx))
     data["test-rmse"] =sqrt(mean_squared_error(testy, model.predict(testx)))
     data["test-r2"] = metrics.r2_score(testy, model.predict(testx))
     return data

def eval_ridge2(trainx, trainy, valx, valy, testx, testy, alpha):
     comb_datax = np.concatenate((trainx, valx))
     comb_datay = np.concatenate((trainy, valy))
     model = Ridge(alpha = alpha)
     data = {}
     model.fit(comb_datax, comb_datay)
     data["train-rmse"] =sqrt(mean_squared_error(trainy, model.predict(trainx)))
     data["train-r2"] = metrics.r2_score(trainy, model.predict(trainx))
     data["val-rmse"] =sqrt(mean_squared_error(valy, model.predict(valx)))
     data["val-r2"] = metrics.r2_score(valy, model.predict(valx))
     data["test-rmse"] =sqrt(mean_squared_error(testy, model.predict(testx)))
     data["test-r2"] = metrics.r2_score(testy, model.predict(testx))
     return data
     

def eval_lasso2(trainx, trainy, valx, valy, testx, testy, alpha):
     comb_datax = np.concatenate((trainx, valx))
     comb_datay = np.concatenate((trainy, valy))
     model = Lasso(alpha = alpha)
     data = {}
     model.fit(comb_datax, comb_datay)
     data["train-rmse"] =sqrt(mean_squared_error(trainy, model.predict(trainx)))
     data["train-r2"] = metrics.r2_score(trainy, model.predict(trainx))
     data["val-rmse"] =sqrt(mean_squared_error(valy, model.predict(valx)))
     data["val-r2"] = metrics.r2_score(valy, model.predict(valx))
     data["test-rmse"] =sqrt(mean_squared_error(testy, model.predict(testx)))
     data["test-r2"] = metrics.r2_score(testy, model.predict(testx))
     return data