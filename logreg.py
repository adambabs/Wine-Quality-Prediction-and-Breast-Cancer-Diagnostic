import numpy as np

class LogReg(object):
    
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
    
    def sigmoid_function(self, w, x):
        a = np.dot(x, w) # numpy dot product transposes automatically
        sigmoid = 1 / (1 + np.exp(-a))
        
        return sigmoid
    
    def gradient_descent(self, sigmoid_result, x, y):
        gd = np.dot(x.T, (y-sigmoid_result))
        
        return gd
    
    def update_weight(self, w, gradient, learning_rate):
        updated_weight = w + (learning_rate*gradient)

        return updated_weight
    
    def cross_entropy_loss(self, sigmoid_result, y):
        eps = 1e-20 # to make sure there are no log(0)
        y1 = np.dot(y.T, np.log(sigmoid_result + eps))
        y0 = np.dot((1-y).T, np.log(1-sigmoid_result + eps))
        loss = -(y1+y0)
        
        return np.asscalar(loss)
    
    def add_bias(self, x):
        bias = np.ones((x.shape[0], 1))
        x = np.concatenate((bias, x), axis=1)
        
        return x
    
    def fit(self, x, y):
        bias = np.ones((x.shape[0], 1))
        x = self.add_bias(x)
        w = np.zeros((x.shape[1],1))
        
        for i in range(self.iterations):
            sigmoid_result = self.sigmoid_function(w, x)
            iter_loss = self.cross_entropy_loss(sigmoid_result, y)
            self.loss.append(iter_loss)
            gradient = self.gradient_descent(sigmoid_result, x, y)
            w = self.update_weight(w, gradient, self.learning_rate)
            
        self.w = w

        return True
    
    def predict(self, x):
        x = self.add_bias(x)
        sigmoid_squash = self.sigmoid_function(self.w, x)
        for i in range(sigmoid_squash.shape[0]):
            if sigmoid_squash[i][0] > 0.5:
                sigmoid_squash[i][0] = 1.0
            else:
                sigmoid_squash[i][0] = 0.0
        
        self.y_predicted = sigmoid_squash
        
        return sigmoid_squash
    
def evaluate_acc(y, y_predicted):
    sum_same = sum((y==y_predicted))
    perc = sum_same/y.shape[0]
    
    return perc[0]

def minmax_normalization(x):
    for i in range(x.shape[1]):
        col_min = np.min(x[:,i])
        col_max = np.max(x[:,i])
        x[:,i] = (x[:,i] - col_min) / (col_max - col_min)
        
    return x

def zscore_normalization(x):
    for i in range(x.shape[1]):
        col_mean = np.mean(x[:,i])
        col_std = np.std(x[:,i])
        
        x[:,i] = (x[:,i] - col_mean) / col_std
        
    return x

def log_transform(x):
    for i in range(x.shape[1]):
        x[:,i] = np.log(x[:,i])
        
    return x

