from collections import Counter
import numpy as np


def read_file(filename):
    emails = list()
    label = list()
    with open(filename, 'r') as f:
        for line in f:
            row = line.strip().split()
            label.append(int(row[0]))
            emails.append(row[1:])
    return emails, label


def build_vocab(train, test, minn):
    vocabulary = Counter()
    for email in train:
        vocabulary.update(email)  
    for email in test:
        vocabulary.update(email)
        
    vocab = [w for w in vocabulary if vocabulary[w] >= minn]
  
    vocab_map = {w:i for i,w in enumerate(vocab)}

    train_vecs = []
    for email in train:
        vec = [0] * len(vocab)
        for word in email:
            if word in vocab_map:
                vec[vocab_map[word]] = 1
        train_vecs.append(vec)

    test_vecs = []
    for email in test:
        vec = [0] * len(vocab)
        for word in email:
            if word in vocab_map:
                vec[vocab_map[word]] = 1
        test_vecs.append(vec)
  
    return np.array(train_vecs), np.array(test_vecs), vocab_map
	

class Perceptron():

	def __init__(self, epoch):
		self.epoch = epoch
		self.w = None
		return

	def get_weight(self):
		return self.w

	def sample_update(self, x, y):
		prediction = np.dot(x, self.w)
		mistake = 0
		if y * prediction <= 0:
			mistake = 1
			self.w += y * x
		if prediction == 0:
			mistake = 1
			self.w += x	
		return self.w, mistake
    
	def train(self, trainx, trainy):
		self.w = np.zeros(trainx.shape[1])
		mistakes = {}
		epoch = 0
		while True:
			epoch += 1
			mistakes[epoch] = 0
			for x, y in zip(trainx, trainy):
				# prediction = np.dot(x, self.w)
				# if y * prediction <= 0:
				# 	mistakes[epoch] += 1
				# 	self.w += y * x
				self.w, mis = self.sample_update(x,y)
				mistakes[epoch]+= mis
			if mistakes[epoch] == 0 or epoch == self.epoch:
				break
		return  mistakes

	def predict(self, newx):
		preds = np.dot(newx, self.w)
		preds[preds <= 0] = -1
		preds[preds > 0] = 1
		return preds

class AvgPerceptron(Perceptron):
	
	def __init__(self, epoch):
		super().__init__(epoch)
		self.w_avg = None
		
	def get_weight(self):
		return self.w_avg

	def train(self, trainx, trainy):
		self.w = np.zeros(trainx.shape[1])
		self.w_avg = np.zeros(trainx.shape[1])
		self.n_updates = 0
		epoch = 0
		mistakes = {}
		while True:
			epoch += 1
			mistakes[epoch] = 0
			for x, y in zip(trainx, trainy):
				# prediction = np.dot(x, self.w)
				# if y * prediction <= 0:
				# 	mistakes[epoch] += 1
				# 	self.w += y * x
				self.w, mis = self.sample_update(x,y)
				mistakes[epoch]+= mis
				self.w_avg += self.w
				self.n_updates += 1
			if mistakes[epoch] == 0 or epoch == self.epoch:
				break
		self.w_avg /= self.n_updates
		return mistakes

	def predict(self, newx):
		preds = np.dot(newx, self.w_avg)
		preds[preds <= 0] = -1 
		preds[preds > 0] = 1

		return preds