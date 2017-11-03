import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mnist import MNIST

def Problem3():
	n = 50 #Samples
	d = 75 #Features
	k = 5
	sigma = 1

	#Generate X
	X = np.random.normal(0, 1, (n,d))

	#Generate w
	w = np.arange(d, dtype = np.float32)

	#Generate Y
		#Generate w*
	w_s = np.zeros(d)
	for i in range(k):
		sign = np.random.choice([True, False])
		if sign == True:
			w_s[i] = 10
		else:
			w_s[i] = -10
		#Generate eps
	eps = np.random.normal(0, sigma**2, n)
	Y = np.dot(X, w_s) + eps

	#Run LASSO
	lambda_list = []
	precision = []
	recall = []
	lambda_ = np.max(2.*abs(np.dot(X.transpose(), Y - np.mean(Y))))
	print(lambda_)
	delta = 0.01
	for i in range(64):
		w, w_0 = LASSO(X, Y, w, lambda_, delta)

		#Calculate numner of 0 terms
		correct_nonzero_count = 0
		total_nonzero_count = 0
		for j in range(len(w)):
			if j < 5 and w[j] != 0.:
				correct_nonzero_count += 1
			if w[j] != 0.:
				total_nonzero_count += 1

		if total_nonzero_count != 0:
			lambda_list.append(lambda_)
			precision.append(correct_nonzero_count/float(total_nonzero_count))
			recall.append(correct_nonzero_count/float(k))

		#Update variables for next pass
		lambda_ *= 0.90
		w = np.arange(d, dtype = np.float32)

	plt.plot(lambda_list, precision, c = 'r', label = 'Precision')
	plt.plot(lambda_list, recall, c = 'g', label = 'Recall')
	plt.title('Recall & Precision vs. Lambda')
	plt.xlabel('Lambda')
	plt.ylabel('Recall & Precision')
	plt.legend()
	plt.show()

def Problem4():
	def error(X, Y, W, w_0):
		#Calculate True error
		WX_predicted = np.dot(W,X.transpose())
		sum_val = 0

		for i in range(len(WX_predicted)):
			sum_val += np.dot(Y[i] - (WX_predicted[i] + w_0).transpose(), Y[i] - (WX_predicted[i] + w_0))#add offset

		return float(sum_val)/X.shape[0]

	#Load a text file of strings
	featureNames = open("upvote_features.txt").read().splitlines()

	#Generate X, Load a csv of floats
	X = np.genfromtxt("upvote_data.csv",delimiter=",")

	#(Samples, Features)
	#(6000   ,	 1000)
	#(n	  ,		d)

	#Generate w
	W = np.zeros(1000)

	#Generate Y, Load a text file of integers
	Y = np.loadtxt("upvote_labels.txt",dtype = np.int)

	X = X - np.mean(X, 0)
	#Generate Test and Train Sets
	X_train = X[:4000]
	X_val = X[4000:5000]
	X_test = X[5000:]
	Y_train = Y[:4000]
	Y_val = Y[4000:5000]
	Y_test = Y[5000:]

	lambda_ = np.max(2.*abs(np.dot(X_train.transpose(), Y_train - np.mean(Y_train))))
	delta = 0.1

	lasso_list = []
	lambda_list = []
	train_error = []
	val_error = []
	nonzero_list = []
	for i in range(60):
		W, w_0 = LASSO(X_train, Y_train, W, lambda_, delta)

		lambda_list.append(lambda_)
		train_error.append(error(X_train, Y_train, W, w_0)**2.)
		val_error.append(error(X_val, Y_val, W, w_0)**2.)

		nonzero = 0
		for element in W:
			if element != 0:
				nonzero += 1
		nonzero_list.append(nonzero)

		#print i
		lambda_ *= 0.95

	print("Test Error:", error(X_test, Y_test, W, w_0))

	largest_features = np.argsort(abs(W))[-9:]
	for feature in largest_features:
		print(featureNames[feature], W[feature])

	plt.plot(lambda_list, train_error, c = 'r', label = 'Train')
	plt.plot(lambda_list, val_error, c = 'g', label = 'Val')
	plt.title('Squared Error vs. Lambda')
	plt.xlabel('Lambda')
	plt.ylabel('Train & Val Squared Error')
	plt.legend()
	plt.show()

	plt.plot(lambda_list, nonzero_list)
	plt.title('Nonzero Entries of W vs. Lambda')
	plt.xlabel('Lambda')
	plt.ylabel('Nonzero Entries of W')
	plt.legend()
	plt.show()

	# Test Error: 5.55871006289

	# sqrt(ReviewNumLineBreaks*UserCoolVotes) 9.10138675786
	# ReviewInFall*InGlendale 9.10757242571
	# sq(ReviewNumWords*UserNumReviews) -10.0678964219
	# sqrt(UserFunnyVotes*InPhoenix) 10.8422371914
	# UserUsefulVotes*InScottsdale 11.2788841314
	# ReviewNumCharacters*ReviewNumLineBreaks 11.4433897163
	# sqrt(UserCoolVotes*BusinessNumStars) 15.8139232593
	# ReviewDate*UserNumReviews -25.9219389649
	# sqrt(ReviewNumCharacters*UserCoolVotes) 67.4956490855
#%%
    
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mnist

def Problem5():
	def load_dataset():
		mndata = mnist.MNIST('./python-mnist/data/')
		X_train, labels_train = map(np.array, mndata.load_training())
		X_test, labels_test = map(np.array, mndata.load_testing())
		X_train = X_train/255.0
		X_test = X_test/255.0

		X_train_temp = []
		labels_train_temp = []
		i = 0
		for label in labels_train:
			if label == 2 or label == 7:
				X_train_temp.append(X_train[i])
				labels_train_temp.append(label)
			i += 1

		X_test_temp = []
		labels_test_temp = []
		i = 0
		for label in labels_test:
			if label == 2 or label == 7:
				X_test_temp.append(X_test[i])
				labels_test_temp.append(label)
			i += 1

		return np.array(X_train_temp), np.array(X_test_temp), np.array(labels_train_temp), np.array(labels_test_temp)

	def encode(Y):
		Y_train = np.zeros(Y.shape[0])
		for i in range(Y.shape[0]):
			if Y[i] == 2:
				Y_train[i]= -1
			else:
				Y_train[i]= 1
		return Y_train

	def misclassification(X, Y, W, b):
		predicted = np.sign(b + np.dot(W,X.transpose()))
		error = Y - predicted
		return np.count_nonzero(error)

	def u(X, Y, W,b):
		return 1./(1. + np.exp(-Y*(b + np.dot(W,X.transpose()))))

	def gradient(X, Y, W, b, lambda_):
		dW = (1./X.shape[0])*np.dot(-Y*(1 - u(X, Y, W,b)), X) + 2*lambda_*W
		db = np.mean(-Y*(1 - u(X, Y, W,b)))

		return dW, db

	def hessian(X, Y, W, b, lambda_):
		n, d = np.shape(X)
		ddW = np.zeros([d,d]);
		coeff = u(X, Y, W,b)*(1 - u(X, Y, W,b))

		for i in range(0,n):
			ddW += coeff[i]*np.dot(X[i].transpose(),X[i]) + 2*lambda_
		ddW *= (1./n)

		print(ddW[0])

		#ddW = (1./X.shape[1])*np.dot(X.transpose(),X)*u(X, Y, W,b)*(1 - u(X, Y, W,b)) + 2*lambda_
		ddb = np.mean(u(X, Y, W,b)*(1 - u(X, Y, W,b)))

		return ddW, ddb

	def J(X, Y, W, b, lambda_):
		return np.mean(np.log(1. + np.exp(-Y*(b + np.dot(W,X.transpose()))))) + lambda_*np.linalg.norm(W)**2

	def gradient_decent(X, Y, W, b, lambda_ = 0.1, lr = 0.001, conv_change = 0.001):
		W_list = []
		b_list = []
		cost_list = []
		misclass_list = []

		cost = J(X, Y, W, b, lambda_)
		cost_list.append(cost)
		cost_change = conv_change + 1
		while cost_change > conv_change:
			W_list.append(np.array(W))
			b_list.append(b)

			dW, db = gradient(X, Y, W, b, lambda_)

			W -= lr*dW
			b -= lr*db

			cost_old = cost
			cost = J(X, Y, W, b, lambda_)
			cost_list.append(cost)
			cost_change = cost_old - cost

			misclass_list.append(misclassification(X, Y, W, b))

		return np.array(W_list), np.array(b_list), np.array(cost_list), np.array(misclass_list)

	def stochastic(X, Y, W, b, lambda_ = 0.1, lr = 0.001, conv_change = 0.001, batch_size = 1):
		W_list = []
		b_list = []
		cost_list = []
		misclass_list = []

		cost = J(X, Y, W, b, lambda_)
		cost_list.append(cost)
		cost_change = conv_change + 1

		while abs(cost_change) > conv_change:
		#for i in range(1):
			W_list.append(np.array(W))
			b_list.append(b)

			index = np.arange(X.shape[0])
			np.random.shuffle(index)
			shuffled_index = index[:batch_size]

			dW, db = gradient(X_train[0:shuffled_index[0]], Y_train[0:shuffled_index[0]], W, b, lambda_)

			W -= lr*dW
			b -= lr*db

			cost_old = cost
			cost = J(X, Y, W, b, lambda_)
			cost_list.append(cost)
			cost_change = cost_old - cost

			misclass_list.append(misclassification(X, Y, W, b))

		return np.array(W_list), np.array(b_list), np.array(cost_list), np.array(misclass_list)

	def newtons_method(X, Y, W, b, lambda_ = 0.1, lr = 0.001, conv_change = 0.001):
		W_list = []
		b_list = []
		cost_list = []
		misclass_list = []

		cost = J(X, Y, W, b, lambda_)
		cost_list.append(cost)
		cost_change = conv_change + 1
		while cost_change > conv_change:
			W_list.append(np.array(W))
			b_list.append(b)

			dW, db = gradient(X, Y, W, b, lambda_)
			ddW, ddb = hessian(X, Y, W, b, lambda_)

			vW = np.linalg.solve(ddW, -dW)
			vb = -db/ddb
			W -= lr*vW
			b -= lr*vb

			cost_old = cost
			cost = J(X, Y, W, b, lambda_)
			cost_list.append(cost)
			cost_change = cost_old - cost

			misclass_list.append(misclassification(X, Y, W, b))

		return np.array(W_list), np.array(b_list), np.array(cost_list), np.array(misclass_list)

	X_train, X_test, Y_train, Y_test = load_dataset()

	#Generate X
	X_train = X_train - np.mean(X_train)
	X_test = X_test - np.mean(X_test)

	#Generate Y
	Y_train = encode(Y_train)
	Y_test = encode(Y_test)

	#Generate W
	W = np.zeros(X_train.shape[1])

	#Generate b
	b = 0

	#5b
	#########################################################
	#W_GD, b_GD, cost_GD_train, misclass_GD_train = gradient_decent(X_train, Y_train, W, b, lr = 0.003, conv_change = 0.000004)

	# cost_list_test = []
	# misclass_list_test = []
	# for i in range(len(W_GD)):
	# 	cost_list_test.append(J(X_test, Y_test, W_GD[i], b_GD[i], 0.1))
	# 	misclass_list_test.append(misclassification(X_test, Y_test, W_GD[i], b_GD[i]))
	# cost_GD_test = np.array(cost_list_test)
	# misclass_GD_test = np.array(misclass_list_test)

	# fig = plt.figure()
	# plt.plot(np.arange(len(cost_GD_train)), cost_GD_train,label = 'Train set')
	# plt.plot(np.arange(len(cost_GD_test)), cost_GD_test, label='Test set')
	# plt.xlabel('Iterations')
	# plt.ylabel('Cost value')
	# plt.legend(fontsize=18)
	# plt.show()
	# fig.savefig('q5_1.jpg')

	# fig = plt.figure()
	# plt.plot(np.arange(len(misclass_GD_train)), misclass_GD_train,label='Train set')
	# plt.plot(np.arange(len(misclass_GD_test)), misclass_GD_test,label='Test set')
	# plt.gca().set_xlim(left=-100)
	# plt.xlabel('Number of iterations')
	# plt.ylabel('Number of misclassifications')
	# plt.legend(fontsize=18)
	# plt.show()
	#########################################################

	#5c FINISH
	#########################################################
	W_SGD, b_SGD, cost_SGD_train, misclass_SGD_train = stochastic(X_train, Y_train, W, b, lr = 0.01, conv_change = 0.0000004, batch_size = 1)

	cost_list_test = []
	misclass_list_test = []
	for i in range(len(W_SGD)):
		cost_list_test.append(J(X_test, Y_test, W_SGD[i], b_SGD[i], 0.1))
		misclass_list_test.append(misclassification(X_test, Y_test, W_SGD[i], b_SGD[i]))
	cost_SGD_test = np.array(cost_list_test)
	misclass_SGD_test = np.array(misclass_list_test)

	fig = plt.figure()
	plt.plot(np.arange(len(cost_SGD_train)), cost_SGD_train,label = 'Train set')
	plt.plot(np.arange(len(cost_SGD_test)), cost_SGD_test, label='Test set')
	plt.xlabel('Iterations')
	plt.ylabel('Cost value')
	plt.legend(fontsize=18)
	plt.show()

	# fig = plt.figure()
	# plt.plot(np.arange(len(misclass_GD_train)), misclass_GD_train,label='Train set')
	# plt.plot(np.arange(len(misclass_GD_test)), misclass_GD_test,label='Test set')
	# plt.gca().set_xlim(left=-100)
	# plt.xlabel('Number of iterations')
	# plt.ylabel('Number of misclassifications')
	# plt.legend(fontsize=18)
	# plt.show()
	#########################################################

	#5d FINISH
	#########################################################
	#W_SGD, b_SGD, cost_SGD_train, misclass_SGD_train = stochastic(X_train, Y_train, W, b, lr = 0.003, conv_change = 0.0000004)
	#########################################################

	#5e FINISH
	#########################################################
	# W_NM, b_NM, cost_NM_train, misclass_NM_train = newtons_method(X_train, Y_train, W, b, lr = 0.003, conv_change = 0.0000004)

	# cost_list_test = []
	# misclass_list_test = []
	# for i in range(len(W_GD)):
	# 	cost_list_test.append(J(X_test, Y_test, W_GD[i], b_GD[i], 0.1))
	# 	misclass_list_test.append(misclassification(X_test, Y_test, W_GD[i], b_GD[i]))
	# cost_GD_test = np.array(cost_list_test)
	# misclass_GD_test = np.array(misclass_list_test)

	# fig = plt.figure()
	# plt.plot(np.arange(len(cost_GD_train)), cost_GD_train,label = 'Train set')
	# plt.plot(np.arange(len(cost_GD_test)), cost_GD_test, label='Test set')
	# plt.xlabel('Iterations')
	# plt.ylabel('Cost value')
	# plt.legend(fontsize=18)
	# plt.show()

	# fig = plt.figure()
	# plt.plot(np.arange(len(misclass_GD_train)), misclass_GD_train,label='Train set')
	# plt.plot(np.arange(len(misclass_GD_test)), misclass_GD_test,label='Test set')
	# plt.xlabel('Number of iterations')
	# plt.ylabel('Number of misclassifications')
	# plt.legend(fontsize=18)
	# plt.show()
	#########################################################

def LASSO(X, Y, W, lambda_, delta):
	i = 0
	stop = False
	a_k = 2.*np.linalg.norm(X.transpose(), axis = 1)**2.
	while stop == False:
		stop = True
		W_old = np.array(W)
		w_0 = np.mean(Y - np.dot(W,X.transpose()))
		for k in range(X.shape[1]):
			c_k = 2.*np.dot(X.transpose()[k], Y - (w_0 + np.dot(W, X.transpose()) - np.dot(W[k], X.transpose()[k])))

			if c_k < -lambda_:
				W[k] = (c_k + lambda_)/a_k[k]
			elif c_k > lambda_:
				W[k] = (c_k - lambda_)/a_k[k]
			else:
				W[k] = 0.

		if np.linalg.norm(W - W_old) > delta:
			stop = False

		#print i
		i += 1
	return W, w_0

def main():
	#Problem3()
	#Problem4()
	Problem5()

if __name__ == '__main__':
	main()



#I think I might be on to what our issue is. When classifying you need to include the offset x_0. I tried adding it and while the the val and test error both go down, the test error is always higher even with the offset. Are you supposed to add the offset to BOTH the training and val error calculations or just the training?
