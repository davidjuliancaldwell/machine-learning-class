# import modules
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
sns.set()

v_1 = np.array([[3/5],[4/5]])
v_2 = np.array([[-4/5],[3/5]])
V = np.squeeze(np.array([v_1,v_2])).T
variance = 1
n = 100

params_int_zipped = zip([np.array([[3,4]]).T,
                  np.array([[3,4]]).T,
                  np.array([[3,4]]).T,
                  np.array([[-4,3]]).T,
                  np.array([[0,4]]).T],
                 [1,8,1,8,8],
                 [1,1,8,1,1])
iteration = 1
for params in params_int_zipped:

    w_star = params[0]
    lambda_1 = params[1]
    lambda_2 = params[2]

    x_vec = np.random.randn(2,n)
    # solve for z
    b = np.zeros((1,2))
    n = 100
    x_vec = np.random.randn(n,2)

    b[0,0] = -(1/n)*np.sum(x_vec[:,0])
    b[0,1] = -(1/n)*np.sum(x_vec[:,1])
    x_vec = x_vec + b

    w1,v1 = scipy.linalg.eig(np.dot(x_vec.T,x_vec))
    idx = w1.argsort()[::-1]
    w1 = np.real(w1[idx])
    v1 = v1[:,idx]
    #a_1 = np.dot((np.diag(w1)**-0.5),v1)
    a_1 = np.dot((scipy.linalg.sqrtm(np.linalg.inv(np.diag(w1)))),v1)

    transx_1 = np.dot(x_vec,a_1)


    sigma = np.diag([lambda_1,lambda_2])
    sigma_neghalf = scipy.linalg.sqrtm(np.linalg.inv(sigma))
    sigma_half = scipy.linalg.sqrtm(sigma)

    trans_x = np.dot(transx_1,np.dot(sigma_half,V.T))

    b[0,0] = -(1/n)*np.sum(trans_x[:,0])
    b[0,1] = -(1/n)*np.sum(trans_x[:,1])
    z = trans_x + b

    # confidence ellipsoid

    theta = np.arange(0,2*np.pi,0.001)
    cos_sin = np.array([np.cos(theta),np.sin(theta)])
    v_sig = np.dot(V,sigma_neghalf)
    u = np.dot(v_sig,cos_sin)
    u = u + w_star

    plt.figure(iteration,dpi=600)
    plt.plot(z[:,0],z[:,1],'o',label='z')
    plt.plot(u[0,:],u[1,:],label='confidence ellipsoid')
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.title('wstar=[{},{}], diag(Lambda) = [{},{}]'.format(w_star[0],w_star[1],lambda_1,lambda_2))
    plt.savefig("hw3_prob1_iter_{}".format(iteration))
    print(iteration)
    #plt.show()
    iteration += 1
