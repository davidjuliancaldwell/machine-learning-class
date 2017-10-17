import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import mnist

sns.set()

# data loading function
# data loading function
def load_dataset():
    mndata = mnist.MNIST('./python-mnist/data')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, X_test,labels_train,labels_test

# training function
def train(x,y,lambda_cnst):
    #a = (np.dot(x.T,x)) + lambda_cnst*np.identity(np.shape(x)[1])
    #b = (np.dot(x.T,y))
    x = x.T
    y = y.T
    a = (np.dot(x,x.T)) + lambda_cnst*np.identity(np.shape(x)[0])
    b = (np.dot(x,y.T))
    w_hat = np.linalg.solve(a,b)
    return w_hat

# prediction function
def predict(w,x_prime):
    #classified = np.argmax(np.dot(x_prime,w),axis=1)
    classified = np.argmax(np.dot(w.T,x_prime.T),axis=0)
    return classified

# evaluate performance
def evaluate(results,truth):
    truth_array = np.array(truth)
    results_array = np.array(results)
    accuracy = np.sum(truth_array==results_array)/len(truth_array)
    #import pdb; pdb.set_trace()
    return accuracy

def feature_transform_old(pixels,p):
    var = 0.1
    std_dev = var**0.5
    center = 0
    d = np.shape(pixels)[0]
    g_vec = std_dev*np.random.randn(d,p) + center
    b = np.squeeze(2*np.pi*np.random.rand(p,1))

    transformed_pixels = np.cos(np.squeeze(np.dot(g_vec.T,pixels))+b)
    return transformed_pixels

def feature_transform(data,p):
    # NEED SAME RANDOM SEED
    np.random.seed(1)

    var = 0.1
    std_dev = var**0.5
    center = 0
    d = np.shape(data)[1]
    g_vec = std_dev*np.random.randn(d,p) + center
    b = np.squeeze(2*np.pi*np.random.rand(p,1))
    data_matrix = np.zeros((np.shape(data)[0],p))
    data_matrix = [(np.dot(g_vec.T,row)+b) for row in data]
    transformed_data = np.cos(data_matrix)
    return transformed_data

def random_partition(data,labels_onehot,labels,proportion):
    # randomly generate indices to select training data from
    inds = np.array(np.random.choice(len(data),np.int(proportion*len(data)),replace=False))

    # select the training data and training labels
    train_set = data[inds,:]
    train_onehot_labels = labels_onehot[inds,:]
    train_labels = labels[inds]

    # make sets to figure out the elements not in the training set for the validation set
    inds_set = set(inds)
    total_set = set(np.arange(len(data)))

    # create set that has other indices
    valid_inds = np.array(list(total_set - inds_set))

    # get validation labels and data
    validation_set = data[valid_inds,:]
    validation_onehot_labels = labels_onehot[valid_inds,:]
    validation_labels = labels[valid_inds]
    #import pdb; pdb.set_trace()
    return train_set,train_onehot_labels,train_labels,validation_set,validation_onehot_labels,validation_labels,inds,valid_inds

x_train,x_test,labels_train,labels_test = load_dataset()

###################################

# one hot lable encoding
labels_onehot_train = np.zeros((np.max(np.size(labels_train)),10))
for i in range(len(labels_onehot_train)):
    labels_onehot_train[i,labels_train[i]] = 1

labels_onehot_test = np.zeros((np.max(np.size(labels_test)),10))
for i in range(len(labels_onehot_test)):
    labels_onehot_test[i,labels_test[i]] = 1

# train on training data, test on test set
lambda_cnst = 10**-4
w_hat = train(x_train,labels_onehot_train,lambda_cnst)
predictions_train = predict(w_hat,x_train)
predictions_test = predict(w_hat,x_test)

train_acc = evaluate(predictions_train,labels_train)
test_acc = evaluate(predictions_test,labels_test)

print('The training accuracy is {} %'.format(100*train_acc))
print('The test accuracy is {} %'.format(100*test_acc))

########

# split data

prop_train = 0.8
train_s,train_ohl,train_l,valid_s,valid_ohl,valid_l,train_inds,valid_inds = random_partition(x_train,labels_onehot_train,labels_train,prop_train)

#p_list = [10,100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]
p_list = [10,100,200,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]

train_error_vec = []
valid_error_vec = []

for p in p_list:
    train_s_trans = feature_transform(train_s,p)
    valid_s_trans = feature_transform(valid_s,p)

    w_hat = train(train_s_trans,train_ohl,lambda_cnst)

    predictions_train = predict(w_hat,train_s_trans)

    predictions_valid = predict(w_hat,valid_s_trans)

    train_err = 1 - evaluate(predictions_train,train_l)
    valid_err = 1 - evaluate(predictions_valid,valid_l)

    train_error_vec.append(train_err)
    valid_error_vec.append(valid_err)

fig1 = plt.figure(figsize=(4, 4), dpi=600)
plt.plot(p_list,train_error_vec,label='training error')
plt.plot(p_list,valid_error_vec,label='validation error')
plt.xlabel('p')
plt.ylabel('error')
plt.title('Training and validation error for different values of p')
plt.legend()
fig1.savefig('hw1_prob7_errorPlot')

# hoeffding's , using p from above

# optimal p
p = 6000

train_s_trans = feature_transform(train_s,p)
w_hat = train(train_s_trans,train_ohl,lambda_cnst)

test_trans = feature_transform(x_test,p)

predictions_test = predict(w_hat,test_trans)

test_acc = evaluate(predictions_test,labels_test)
print('The test accuracy using cosine features for p = {}, is {} %'.format(p,100*test_acc))

sigma = 0.05
hoeff = np.sqrt(np.log(2/sigma)/(2*np.shape(test_trans)[0]))

conf_interval = [test_acc-hoeff,test_acc+hoeff]
conf_interval[0]
conf_interval[1]
print('the confidence interval is {}%, {}%'.format(100*conf_interval[0],100*conf_interval[1]))
