#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np

class RNNnp():
    def __init__(self, letter_dim, hidden_dim=200, bptt_truncate=4):
        
        self.bptt_truncate = bptt_truncate
        self.hidden_dim = hidden_dim
        self.letter_dim = letter_dim
        # U = 200x29
        self.U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./letter_dim), (hidden_dim, letter_dim))
        # V = 29x200
        self.V = np.random.uniform(-np.sqrt(1./letter_dim), np.sqrt(1./hidden_dim), (letter_dim, hidden_dim))
        # W = 200x200
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.U, self.V, self.W = load_model_parameters("Canh.npz")
    
    def feedforward(self, X):
        # X input vào là 1 từ có nhiều chữ cái, chúng ta sẽ predict chữ cái có khả năng xuất hiện tiếp theo
        # Lấy số lượng chữ cái có trong X để tạo hidden. Vd: X="madara"
        T = len(X);
        # Phải giữ lại hidden state để lát xài
        # Thêm 1 state tại vì state[0] ko có gì để compute, nên sẽ xài state[-1]
        s = np.zeros((T + 1, self.hidden_dim))
        # Output o sẽ là 1-hot vector độ dài bảng chữ cái 29x1 cho T states
        # dự đoán xem chữ nào tiếp theo có khả năng xuất hiện tiếp theo nhất. Chữ đó sẽ có o[i] lớn nhất.
        # Nhưng chúng ta có T state. Mỗi state là 1 vector predict o[i] -> ma trận o[T][letter_dim]
        o = np.zeros((T, self.letter_dim))
        # Bây giờ, với từng chữ cái X
        # x = 1-hot vector 29x1 -> U*x = U[:, X[x]]
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:, X[t]] + np.dot(self.W, s[t-1]))
            o[t] = softmax(np.dot(self.V, s[t]))
            
        return (o, s)
        
    def calculate_loss(self, X, Y):
        # output ra o sẽ là một một ma trận T state,mỗi state
        # là 1-hot vector dự đoán khả năng xuất hiện của từ tiếp theo
        # Y = 29x1
        T = len(Y)
        o, s = self.feedforward(X)
        y = np.zeros((T, self.letter_dim)) 
        y += o
        loss = 0
        
        for t in np.arange(T):
            if (y[t, Y[t]]>1e-7):
                loss += -1. * np.log(y[t, Y[t]]) 
        return loss
    
    def calculate_total_loss(self, X_train, Y_train):
        T = len(Y_train)
        total_loss = 0
        N = sum(len(Y_i) for Y_i in Y_train)
        for t in  np.arange(T):
            loss = self.calculate_loss(X_train[t], Y_train[t])
            total_loss += loss
        return  1.*total_loss/N
        
    def bptt(self, X, Y):   
        
        T = len(Y)
        o, s = self.feedforward(X)
        dLdV = np.zeros((self.letter_dim, self.hidden_dim))
        dLdU = np.zeros((self.hidden_dim, self.letter_dim))
        dLdW = np.zeros((self.hidden_dim, self.hidden_dim))
        delta_o = o
        delta_o[np.arange(T),Y]-=1.
        
        for t in np.arange(T) [::-1]:
            dLdV += np.outer( delta_o[t], s[t].T )
            delta_t = np.dot( self.V.T, delta_o[t])*(1-(s[t]**2))
            for bptt in np.arange(0, t+1) [::-1]:
                dLdW += np.outer(delta_t, s[bptt-1])
                dLdU[:,X[bptt]] += delta_t
                delta_t = np.dot(self.W.T, delta_t)*(1 - (s[bptt-1]**2))
        
        return [dLdU, dLdV, dLdW]
        
    def sgd_step(self, X, Y, learning_rate):
        
        dLdU, dLdV, dLdW = self.bptt(X,Y)
        self.U -= learning_rate*dLdU
        self.V -= learning_rate*dLdV
        self.W -= learning_rate*dLdW
        
def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def load_model_parameters(path):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    
    print "Loaded model parameters from %s" % (path)
    return U,V,W
    







