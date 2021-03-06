'''	perceptron.py

	Implement a single perceptron (artificial neuron), and train it to solve a logical AND operation.

	Execution: python3 perceptron.py

TODO: 
@author Alex Hinton
@date 05/09/22
'''

import random
import numpy as np
import matplotlib.pyplot as plt
from activation_functions import *

	
# TODO: Complete this function to train a perceptron
def train_perceptron( X, Y, learning_rate=0.5, threshold=0, activation_function=step_activation ):
	''' Train a perceptron to predict the training targets Y given training inputs X. 

	ARGS:
		X: (n,m) ndarray of training inputs, in which each row represents 1 sample and each column represents 1 feature
		Y: (n,1) ndarray of training targets
		learning_rate: float, determines how strongly residuals impact weights during each iteration of training.

	RETURNS:
		W: (m+1,) ndarray of weights for each column of X + the bias term (intercept) 
		Y_pred: (n,1) ndarray of predictions for each sample (row) of X
		mse: float, mean squared error
	'''

	# Create a figure window for tracking mse over time
	fig = plt.figure()
	plt.title( "Perceptron training" )
	plt.xlabel( "epoch" )
	plt.ylabel( "MSE" )
	plt.grid( True )


	# Horizontally stack a columne of ones (for the bias term) onto the right side of X
	X = np.hstack((X,np.ones((X.shape[0],1))))

	# Initialize weights (including the bias) to small random numbers
	w = (X.shape[1])
	W = np.ones((w,1))
	for n in range(w):
		r = random.random()
		W[n,0] = r

	print("Initialised weights and bias randomly:")
	print(W)

	# Initialising the current iteration, the iteration cap, and the mse
	# Value of mse here does not matter, only that it's greater than the threshold so the loop triggers for the first time
	maxEpochs = 100
	epoch = 0
	mse = 1

	# Initialising Y_pred here so that it can persist outside the loop
	Y_pred = np.zeros((X.shape[0],1))

	# Loop over training set until error is acceptably small, or iteration cap is reached	
	while (mse > 0.01) & (epoch < maxEpochs):
		for n in range(X.shape[0]):
			xs = X[n,:]
			Y_pred[n] = activation_function(xs, W, threshold)
			for m in range(w):
				W[m] = W[m] - xs[m]*(Y_pred[n] - Y[n])*learning_rate

		mse = np.mean((Y_pred - Y)**2)
		epoch += 1
		
	return W, Y_pred, mse


def test_logical_and():
	''' Train a perceptron to perform a logical AND operation. '''
	print( "\nTESTING LOGICAL AND" )
	truth_table = np.array( [[0,0,0], [0,1,0], [1,0,0], [1,1,1]] )
	n = truth_table.shape[0]
	X = truth_table[:, 0:2]
	Y = truth_table[:, 2].reshape((n,1))
	learning_rate = 0.5
	W, Y_pred, mse = train_perceptron( X, Y, learning_rate, activation_function=step_activation )

	# Display results in the temrinal
	print( "\nWeights:", W.T )
	print( "MSE:", mse )
	print( "------------------" )
	print( " X0  X1 Y  Y* " )
	print( "------------------" )
	print( np.hstack((truth_table, Y_pred)) )
	print( "------------------" )
	
	plt.title( "AND" )
	plt.show()


def test_logical_or():
	''' Train a perceptron to perform a logical OR operation. '''
	print( "\nTESTING LOGICAL OR" )
	truth_table = np.array( [[0,0,0], [0,1,1], [1,0,1], [1,1,1]] )
	n = truth_table.shape[0]
	X = truth_table[:, 0:2]
	Y = truth_table[:, 2].reshape((n,1))
	learning_rate = 0.5 #0.005 is good for ReLU, softplus
	W, Y_pred, mse = train_perceptron( X, Y, learning_rate, activation_function=step_activation )
	
	# Display results in the temrinal
	print( "\nWeights:", W.T )
	print( "MSE:", mse )
	print( "------------------" )
	print( " X0  X1 Y  Y* " )
	print( "------------------" )
	print( np.hstack((truth_table, Y_pred)) )
	print( "------------------" )

	plt.title( "OR" )
	plt.show()


def test_logical_xor():
	''' Train a perceptron to perform a logical XOR operation. '''
	print( "\nTESTING LOGICAL XOR" )
	truth_table = np.array( [[0,0,0], [0,1,1], [1,0,1], [1,1,0]] )
	n = truth_table.shape[0]
	X = truth_table[:, 0:2]
	Y = truth_table[:, 2].reshape((n,1))
	learning_rate = 0.5
	W, Y_pred, mse = train_perceptron( X, Y, learning_rate, activation_function=step_activation )
	
	# Display results in the temrinal
	print( "\nWeights:", W.T )
	print( "MSE:", mse )
	print( "------------------" )
	print( " X0  X1 Y  Y* " )
	print( "------------------" )
	print( np.hstack((truth_table, Y_pred)) )
	print( "------------------" )

	plt.title( "XOR" )
	plt.show()


def test_line():
	''' Train a perceptron to recreate a straight line. '''
	print( "\nTESTING STRAIGHT LINE" )
	n = 50
	X = np.linspace( -10, 10, n ).reshape((n,1))
	m = (np.random.random() - 0.5) * 20
	b = (np.random.random() - 0.5) * 20
	Y = m*X + b
	learning_rate = 0.05
	W, Y_pred, mse = train_perceptron( X, Y, learning_rate, activation_function=linear_activation )
	plt.title( "LINE TEST" )
	
	# Display results in the temrinal
	print( "\nWeights:", W.T )
	print( "MSE:", mse )
	print( "-------------------------" )
	print( " X        Y       Y* " )
	print( "-------------------------" )
	print( np.hstack((X, Y, Y_pred)) )
	print( "-------------------------" )

	plt.figure()
	plt.plot( X, Y, 'ob', alpha=0.5, label="Y" )
	plt.plot( X, Y_pred, 'xk', label="Y*" )
	plt.grid( True )
	plt.xlabel( "X" )
	plt.ylabel( "Y" )
	plt.title( f"Target:  Y  = {m}*X + {b}\nLearned: Y* = {W[0]}*X + {W[1]}" )
	plt.show()


if __name__=="__main__":
	test_logical_and()
	#test_logical_or()
	#test_logical_xor()
	#test_line()