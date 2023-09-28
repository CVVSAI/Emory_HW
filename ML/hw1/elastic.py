
import numpy as np

def loss(x, y, beta, el, alpha):
    N = x.shape[0]
    y_pred = x @ beta
    p1 = np.sum((y - y_pred)**2) / (2 * N)
    p2 = (alpha * np.sum(beta**2) + (1 - alpha) * np.sum(np.abs(beta))) * (el / 2)
    return p1+p2

def grad_step(x, y, beta, el, alpha, eta):
    N = x.shape[0]
    y_pred = x @ beta 
    mse = (2/N) * x.T @ (y_pred - y)
    l2_grad = 2 * el * alpha * beta
    l1_grad = np.sign(beta)
    l1_grad[beta > el * (1 - alpha)] -= (1 - alpha) 
    l1_grad[beta < - el * (1 - alpha)] += (1 - alpha)
    grad = mse+ l2_grad + l1_grad
    beta = beta - eta * grad
    return beta

class ElasticNet:
    def __init__(self, el, alpha, eta, batch, epoch):
        self.beta = None
        self.el = el
        self.alpha = alpha
        self.eta = eta
        self.batch = batch
        self.epoch = epoch

    def coef(self):
        return self.beta

    def train(self, x, y):
        loss_dict = {}
        self.beta = np.zeros((x.shape[1],1))
        for i in range(1, self.epoch+1):
            self.beta = grad_step(x, y, self.beta, self.el, self.alpha, self.eta)
            loss_dict[i] = loss(x, y, self.beta, self.el, self.alpha)
        return loss_dict

    def predict(self, x):
        return np.dot(x ,self.beta)
