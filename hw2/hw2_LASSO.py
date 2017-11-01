# import modules
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
sns.set()

###############

# Functions!

# data generation function
def generateData():
    n = 50
    d = 75
    #n = 4000
    #d = 6000
    k = 5
    sigma = 1

    # generate x_i
    cov_mat = np.identity(d)
    avg = np.zeros((d,))
    x = np.random.multivariate_normal(avg,cov_mat,n)

    # epsilon vec
    epsilon_vec = np.random.normal(loc=0,scale=sigma)

    # w_star_0
    w_star_0 = 0
    w_star = np.zeros((d,))
    w_star[0:k] = np.random.randint(low = 1,high=3, size=k)
    w_star[w_star==2] = -10
    w_star[w_star==1] = 10


    # generate y
    y = np.dot(x,w_star) + w_star_0 + epsilon_vec

    return y,x,w_star,k

# calculate what lambda max should be
def lambda_max_func(x,y):
    k = np.shape(x)[1]
    n = np.shape(x)[0]
    sum_y = (1/n)*np.sum(y[:])
    y_reg = y - sum_y
    k_range = np.arange(k)
    lambda_max = np.max([2*np.abs(np.dot(x[:,k_ind].T,y_reg)) for k_ind in k_range])
    print('the max lambda is {}'.format(lambda_max))
    return lambda_max


### lasso function
def lasso_CD(x,y,lambda_val,w_init):
    ### initialize conditions
    delta = 1e-4
    n = np.shape(x)[0]
    d = np.shape(x)[1]

    stop_cond = 100
    w = np.array(w_init)
    a = np.zeros((d,))
    c = np.zeros((d,))
    stop_cond_vec = []
    w_old = np.zeros((d,))

    while(stop_cond>delta):
        diff_y_x = (y - np.dot(x,w))
        w_0 = (1/n)*np.sum(diff_y_x)
        for k in np.arange(d):
            a[k] = 2*np.dot(x[:,k].T,x[:,k])

            # subtract version

            diff_y_x_w0 = y - w_0 - np.dot(x,w) + x[:,k]*w[k]

            # bool version
#             mask = np.ones(d,dtype=bool)
#             mask[k] = 0
#             x_temp = x[:,mask]
#             w_temp = w[mask]

#             diff_y_x_w0 = y - w_0 - np.dot(x_temp,w_temp)
            c[k] = 2*np.dot(x[:,k].T,diff_y_x_w0)

            if c[k] < -lambda_val:
                w[k] = ((c[k]+lambda_val))/a[k]
            elif (np.abs(c[k]) < lambda_val):
                w[k] = 0
            elif c[k] > lambda_val:
                w[k] = ((c[k]-lambda_val))/a[k]

        weight_diff = w - w_old
        w_old = np.array(w)


        stop_cond = np.max(weight_diff)
        stop_cond_vec.append(stop_cond)

    return w,stop_cond_vec

# loop function which calls lasso
def lasso_loop(x,y,k):
    w_vec = []
    num_non_zero_vec = []
    correct_non_zero_vec = []
    lambda_val = lambda_max_func(x,y)
    lambda_vec = []

    ground_truth = np.zeros((np.shape(x)[1],))
    ground_truth[0:5] = 1
    correct_elems = 100

    w_init = np.random.random(np.shape(x)[1])

    while (correct_elems>(33)):
        w,stop_cond_vec = lasso_CD(x,y,lambda_val,w_init)
        # total non zero
        num_non_zero = sum((np.abs(x)>0 for x in w))

        # correct non zeros
        w_logical = np.abs(w)>0
        correct_non_zero = sum(np.logical_and(ground_truth,w_logical))

        #w_init = np.array(w)
        w_init = np.random.random(np.shape(x)[1])

        w_vec.append(w)
        num_non_zero_vec.append(num_non_zero)
        correct_non_zero_vec.append(correct_non_zero)

        lambda_vec.append(lambda_val)

        ratio_val = 4/5

        lambda_val*=ratio_val
        correct_elems = sum(w_logical == ground_truth)
        print('lambda is {}'.format(lambda_val))

        print('number of correct elements is {}'.format(correct_elems))

    return w_vec,num_non_zero_vec,correct_non_zero_vec,lambda_vec

# plot precission and recall on the same graph
def plot_precision_recall(num_non_zero_vec,correct_non_zero_vec,lambda_vec,k):


    # plot precision and recall
    # count nonzero

    ax1 = plt.figure(figsize=(4, 4), dpi=600)
    ax1 = plt.figure()
    precision = np.array(correct_non_zero_vec)/np.array(num_non_zero_vec)
    plt.semilogx(lambda_vec,precision,label='precision')
    plt.xlabel(r'$ \lambda $')
    recall = np.array(correct_non_zero_vec)/k
    plt.semilogx(lambda_vec,recall,label='recall')
    plt.ylabel('Normalized Precision/Recall')
    plt.title('Precision and Recall vs. Lambda for lasso')
    plt.legend()
    plt.savefig('hw2_prob3_recall_precision_lasso_random')

# plot the lasso values as necessary
def plot_lasso(w,stop_cond_vec,w_star):
    ax1 = sns.distplot(w, rug=True,hist=True,bins=100, kde=False, rug_kws={"color": "g"},hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"})

    plt.figure()
    ax3 = plt.bar(np.arange(len(w_star)),w_star,label='w*')
    ax2 = plt.bar(np.arange(len(w)),1.1*w,label='computed weight matrix')
    plt.legend()
    plt.ylim([-10,10])

    plt.figure()
    ax3 = plt.plot(np.arange(len(stop_cond_vec)),stop_cond_vec)
    plt.ylim([0,0.05])


    ######### Processing

    # generate data
y,x,w_star,k = generateData()

# calculate the weight vector, number of non zeros, and lambda values
w_vec,num_non_zero_vec,correct_non_zero_vec,lambda_vec = lasso_loop(x,y,k)

# plot the precission and recall
plot_precision_recall(num_non_zero_vec,correct_non_zero_vec,lambda_vec,k)
