#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np
import WordProcessing
import PWNet
from datetime import datetime

def train_with_sgd(model, X_train, Y_train, learning_rate=0.0001, nepoch=1000, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_total_loss(X_train, Y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Examples_seen=%d Epoch=%d: Loss = %f Rate=%f" % (time, num_examples_seen, epoch, loss,learning_rate)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                print "Setting learning rate to %f" % learning_rate
                break
            if (len(losses)>1 and losses[-1][1] <= losses[-2][1]):
                save_model_parameters("Canh", model)
#            save_model_parameters("Canh", model)
        # For each training example...
        for i in range(len(Y_train)):
            # One SGD step
            model.sgd_step(X_train[i], Y_train[i], learning_rate)
            num_examples_seen += 1

            
def save_model_parameters(outfile, model):
    U, V, W = model.U, model.V, model.W
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile
    
p = WordProcessing.Process()
X_train, Y_train = p.giveInput()
model = PWNet.RNNnp(29)
train_with_sgd(model, X_train, Y_train)













