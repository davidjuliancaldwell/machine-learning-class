
########################################################################
### import modules
# %%
from __future__ import division
from __future__ import print_function

from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
sns.set()


########################################################################
# %%

### define functions

def f_hat_compute(K,alpha_hat):
    f_hat = np.dot(alpha_hat,K)

    return f_hat

def LOOCV_rbf(x,y,lambda_vec,gamma_vec,loss_func,D):

    cost_vec = np.zeros((lambda_vec.shape[0],gamma_vec.shape[0]))

    for lambda_iter, lambda_val in enumerate(lambda_vec):

        for gamma_iter, gamma_ind in enumerate(gamma_vec):
            print('lambda = {},gamma = {}'.format(lambda_val,gamma_ind))


            score_ind = []

            for valid_ind in np.arange(y.shape[0]):
                mask = np.ones(np.shape(y))
                mask[valid_ind] = 0
                mask = mask.astype('bool')
                y_train = y[mask]
                y_test = y[valid_ind]
                x_train = x[mask]
                x_test = x[valid_ind]
                n = x_train.shape[0]


                K = make_K_mat_rbf(x_train,gamma_ind)
                ################### cvxpy part
                # Construct the problem.
                alpha = Variable(n)
                    #import pdb;pdb.set_trace()

                if loss_func == 'least_squares':
                    cost = sum_squares(y_train - K*alpha) + lambda_val*quad_form(alpha,K)
                    objective = Minimize(cost)
                    prob = Problem(objective)

                elif loss_func == 'huber':
                    cost = sum_entries(huber(y_train - K.T*alpha, 0.01)) + lambda_val*quad_form(alpha,K)
                    objective = Minimize(cost)
                    prob = Problem(objective)

                elif loss_func == 'non_decrease':
                    cost = sum_squares(y_train - K*alpha) + lambda_val*quad_form(alpha,K)
                    objective = Minimize(cost)
                    constraints = [np.dot(D,K)*alpha >= 0]
                    prob = Problem(objective,constraints)


                result = prob.solve()
                alpha_hat = alpha.value
                ####################


                eval_kernel = kernel_eval_rbf(gamma_ind,x_test,x_train)
                predict_y = np.array([np.sum(alpha_hat.A1.T*eval_kernel)])
                cost_ind = cost_func(predict_y,y_test)
                score_ind.append(cost_ind)


            score = (1/len(score_ind))*np.sum(score_ind)
            cost_vec[lambda_iter,gamma_iter] = score

    return cost_vec

def LOOCV_rbf_TV(x,y,D,lambda1_vec,lambda2_vec,gamma_vec):

    cost_vec = np.zeros((lambda1_vec.shape[0],lambda2_vec.shape[0],gamma_vec.shape[0]))

    for lambda1_iter, lambda1_val in enumerate(lambda1_vec):

        for lambda2_iter,lambda2_val in enumerate(lambda2_vec):

            for gamma_iter, gamma_ind in enumerate(gamma_vec):
                print('lambda1 = {},lambda2 = {},gamma = {}'.format(lambda1_val,lambda2_val,gamma_ind))


                score_ind = []

                for valid_ind in np.arange(y.shape[0]):
                    mask = np.ones(np.shape(y))
                    mask[valid_ind] = 0
                    mask = mask.astype('bool')
                    y_train = y[mask]
                    y_test = y[valid_ind]
                    x_train = x[mask]
                    x_test = x[valid_ind]
                    n = x_train.shape[0]


                    K = make_K_mat_rbf(x_train,gamma_ind)
                    ################### cvxpy part
                    # Construct the problem.
                    alpha = Variable(n)
                    #import pdb;pdb.set_trace()

                    cost = sum_squares(y_train - K*alpha) + lambda1_val*norm((np.dot(D,K)*alpha),1) + lambda2_val*quad_form(alpha,K)

                    objective = Minimize(cost)
                    prob = Problem(objective)
                    result = prob.solve()
                    alpha_hat = alpha.value
                    ####################


                    eval_kernel = kernel_eval_rbf(gamma_ind,x_test,x_train)
                    predict_y = np.array([np.sum(alpha_hat.A1.T*eval_kernel)])
                    cost_ind = cost_func(predict_y,y_test)
                    score_ind.append(cost_ind)


                score = (1/len(score_ind))*np.sum(score_ind)
                cost_vec[lambda1_iter,lambda2_iter,gamma_iter] = score

    return cost_vec

def cost_func(prediction,truth):

    cost = []

    for i in np.arange(prediction.shape[0]):

        cost.append(np.linalg.norm(prediction[i]-truth[i]))

    cost_total = np.sqrt(np.sum(cost))
    return cost_total

def make_K_mat_rbf(x,gamma):
    K = np.zeros((x.shape[0],x.shape[0]))

    for i in np.arange(x.shape[0]):
        for j in np.arange(x.shape[0]):
            K[i][j] = np.exp(-gamma*(np.linalg.norm(x[i]-x[j])**2))

    return K

def kernel_eval_rbf(gamma,x_1,x_2):
    eval_kernel = []

    for i in np.arange(x_2.shape[0]):
        eval_iter = float(np.exp(-gamma*(np.linalg.norm(x_1-x_2[i])**2)))
        eval_kernel.append(eval_iter)
    return eval_kernel

def alpha_hat_compute(K,lambda_val,y):

    lhs = K + lambda_val*np.identity(np.shape(K)[0])
    alpha_hat = np.linalg.solve(lhs,y)

    return alpha_hat

def choose_gamma(x):
    dist_vals = []
    for i in np.arange(x.shape[0]):
        for j in np.arange(i):
            dist_vals.append((np.linalg.norm(x[i] - x[j])**2))

    median_val = np.median(np.array(dist_vals))
    gamma = 1/median_val
    return gamma

def plot_function(x,y,x_cont,f_x,f_hat,loss_func,lambda_val,gamma):
    plt.figure(dpi=600)
    #plt.figure()
    plt.plot(x,y,'o',label='original data with noise')
    plt.plot(x_cont,f_x,label='true f(x)')
    plt.plot(x_cont,f_hat,label='f_hat(x)')
    plt.xlabel('x')
    plt.ylabel('f(x) or y')
    plt.title('Data, f(x), and f_hat_x \n for loss function = {}, lambda = {},\n \
        gamma = {:.2f}'.format(loss_func,lambda_val,gamma))
    plt.legend()
    plt.ylim([-5,50])
    #plt.show
    plt.savefig("hw4_prob1_func_{}".format(loss_func))

def plot_function_TV(x,y,x_cont,f_x,f_hat,loss_func,lambda1_val,lambda2_val,gamma):
    plt.figure(dpi=600)
    #plt.figure()
    plt.plot(x,y,'o',label='original data with noise')
    plt.plot(x_cont,f_x,label='true f(x)')
    plt.plot(x_cont,f_hat,label='f_hat(x)')
    plt.xlabel('x')
    plt.ylabel('f(x) or y')
    plt.title('Data, f(x), and f_hat_x \n for loss function = {}, lambda1 = {},\n \
        lambda2 = {}, gamma = {:.2f}'.format(loss_func,lambda1_val,lambda2_val,gamma))
    plt.legend()
    plt.ylim([-5,50])
    #plt.show

    plt.savefig("hw4_prob1_func_{}".format(loss_func))

########################################################################
# %%

### generate data

n = 30
x_i = np.array([(i-1)/(n-1) for i in np.arange(n)])
eps_i = np.random.randn(n,1)

k= np.array([1,2,3,4])

f_x = np.zeros((x_i.shape[0],1))
index = 0
for val in x_i:
    f_x[index] =  10*np.sum(val>(k/5))
    index+=1

y = f_x + eps_i
y[14] = 0
########################################################################
# %%
# generate true f(x)
n_cont = 1000
x_i_cont = np.array([(i-1)/(n_cont-1) for i in np.arange(n_cont)])

f_x_cont = np.zeros((x_i_cont.shape[0],1))
index = 0
for val in x_i_cont:
    f_x_cont[index] =  10*np.sum(val>(k/5))
    index+=1
########################################################################
# %%

# generate d
D = np.zeros((n-1,n))

for i in np.arange(D.shape[0]):
    for j in np.arange(D.shape[1]):
        if i == j:
            D[i,j] = -1.0

        elif i == j - 1:
            D[i,j] = 1.0

x_i = np.expand_dims(x_i,axis=1)
#y = np.expand_dims(y,axis=1)
#f_x_cont = np.expand_dims(f_x_cont,axis=1)
x_i_cont = np.expand_dims(x_i_cont,axis=1)

D_LOOCV = np.zeros((n-2,n-1))

for i in np.arange(D.shape[0]):
    for j in np.arange(D.shape[1]):
        if i == j:
            D[i,j] = -1.0

        elif i == j - 1:
            D[i,j] = 1.0


########################################################################
# %%

### least squares, huber, domain knowledge
loss_funcs = ['least_squares','huber','non_decrease']
#loss_funcs = ['non_decrease']

for loss_func in loss_funcs:
    lambda_vec = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1,10])

    gamma = choose_gamma(x_i)

    gamma_vec = np.arange(gamma-10,gamma+40,5)
    #loss_func = 'least_squares'

    score_vec = LOOCV_rbf(x_i,y,lambda_vec,gamma_vec,loss_func,D_LOOCV)

    best_ind = np.unravel_index(score_vec.argmin(), score_vec.shape)

    best_gamma = gamma_vec[best_ind[1]]
    best_lambda = lambda_vec[best_ind[0]]

    f_hat = np.zeros((x_i_cont.shape))

    K = make_K_mat_rbf(x_i,best_gamma)
    alpha = Variable(n)

    if loss_func == 'least_squares':
        cost = sum_squares(y - K*alpha) + best_lambda*quad_form(alpha,K)
        objective = Minimize(cost)
        prob = Problem(objective)

    elif loss_func == 'huber':
        cost = sum_entries(huber(y - K.T*alpha, 0.01)) + best_lambda*quad_form(alpha,K)
        objective = Minimize(cost)
        prob = Problem(objective)

    elif loss_func == 'non_decrease':
        cost = sum_squares(y - K*alpha) + best_lambda*quad_form(alpha,K)
        objective = Minimize(cost)
        constraints = [np.dot(D,K)*alpha >= 0]
        prob = Problem(objective,constraints)


    result = prob.solve()
    alpha_hat = alpha.value

    for i in np.arange(x_i_cont.shape[0]):
        eval_kernel = kernel_eval_rbf(best_gamma,float(x_i_cont[i]),x_i)
        predict_y = np.sum(alpha_hat.A1.T*np.array(eval_kernel))
        f_hat[i] = predict_y



    plot_function(x_i,y,x_i_cont,f_x_cont,f_hat,loss_func,best_lambda,best_gamma)

########################################################################
# %%
### TV
loss_func = 'total_var'
lambda1_vec = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1,10])
lambda2_vec = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1,10])
gamma = choose_gamma(x_i)

gamma_vec = np.arange(gamma-10,gamma+40,5)

#loss_func = 'least_squares'

score_vec = LOOCV_rbf_TV(x_i,y,D_LOOCV,lambda1_vec,lambda2_vec,gamma_vec)

best_ind = np.unravel_index(score_vec.argmin(), score_vec.shape)

best_gamma = gamma_vec[best_ind[2]]
best_lambda1 = lambda1_vec[best_ind[0]]
best_lambda2 = lambda2_vec[best_ind[1]]

f_hat = np.zeros((x_i_cont.shape))

K = make_K_mat_rbf(x_i,best_gamma)
alpha = Variable(n)

### cost function

cost = sum_squares(y - K*alpha) + best_lambda1*norm((np.dot(D,K)*alpha),1) + best_lambda2*quad_form(alpha,K)

objective = Minimize(cost)
prob = Problem(objective)
result = prob.solve()
alpha_hat = alpha.value

for i in np.arange(x_i_cont.shape[0]):
    eval_kernel = kernel_eval_rbf(best_gamma,float(x_i_cont[i]),x_i)
    predict_y = np.sum(alpha_hat.A1.T*np.array(eval_kernel))
    f_hat[i] = predict_y


plot_function_TV(x_i,y,x_i_cont,f_x_cont,f_hat,loss_func,best_lambda1,best_lambda2,best_gamma)
########################################################################

# %%
### plot a whole bunch
plotMany = False
if plotMany:

    #lambda_vec = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
    lambda_vec = [1e-6,1e-4,1e-2]
    gamma_vec = [5,10,20,40]


    loss_funcs = ['least_squares','huber','non_decrease']
    #loss_funcs = ['non_decrease']

    for loss_func in loss_funcs:

        for lambda_val in lambda_vec:

            for gamma_val in gamma_vec:

                alpha = Variable(n)
                K = make_K_mat_rbf(x_i,gamma_val)


                if loss_func == 'least_squares':
                    cost = sum_squares(y - K*alpha) + lambda_val*quad_form(alpha,K)
                    objective = Minimize(cost)
                    prob = Problem(objective)

                elif loss_func == 'huber':
                    cost = sum_entries(huber(y - K.T*alpha, 0.01)) + lambda_val*quad_form(alpha,K)
                    objective = Minimize(cost)
                    prob = Problem(objective)

                elif loss_func == 'non_decrease':
                    cost = sum_squares(K*alpha - y) + lambda_val*quad_form(alpha,K)
                    objective = Minimize(cost)
                    constraints = [(0 <= (np.dot(D,K)*alpha))]
                    prob = Problem(objective,constraints)

                result = prob.solve()
                alpha_hat = alpha.value

                f_hat = np.zeros((x_i_cont.shape))

                for i in np.arange(x_i_cont.shape[0]):
                    eval_kernel = kernel_eval_rbf(gamma_val,float(x_i_cont[i]),x_i)
                    predict_y = np.sum(alpha_hat.A1.T*np.array(eval_kernel))
                    f_hat[i] = predict_y

                plot_function(x_i,y,x_i_cont,f_x_cont,f_hat,loss_func,lambda_val,gamma_val)
########################################################################

# %%
### plot a whole bunch of TV
if plotMany:
    loss_func = 'total_var'
    lambda1_vec = np.array([1e-5,1e-3,1e-1,1,10,100])
    lambda2_vec = np.array([1e-5])
    gamma = choose_gamma(x_i)

    #gamma_vec = np.arange(gamma-10,gamma+30,10)
    gamma_vec = np.array([50])

    #loss_func = 'least_squares'
    for lambda1_val in lambda1_vec:

        for lambda2_val in lambda2_vec:

            for gamma_val in gamma_vec:
                print('lambda1 = {},lambda2 = {},gamma = {}'.format(lambda1_val,lambda2_val,gamma_val))

                f_hat = np.zeros((x_i_cont.shape))
                K = make_K_mat_rbf(x_i,gamma_val)
                alpha = Variable(n)
                ### cost function
                cost = sum_squares(y - K*alpha) + lambda1_val*norm((np.dot(D,K)*alpha),1) + lambda2_val*quad_form(alpha,K)

                objective = Minimize(cost)
                prob = Problem(objective)
                result = prob.solve()
                alpha_hat = alpha.value

                for i in np.arange(x_i_cont.shape[0]):
                    eval_kernel = kernel_eval_rbf(gamma_val,float(x_i_cont[i]),x_i)
                    predict_y = np.sum(alpha_hat.A1.T*np.array(eval_kernel))
                    f_hat[i] = predict_y

                plot_function_TV(x_i,y,x_i_cont,f_x_cont,f_hat,loss_func,lambda1_val,lambda2_val,gamma_val)
