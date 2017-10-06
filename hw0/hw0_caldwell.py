# import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# part A
n_ind = 40000
plt.figure()
Z = np.random.randn(n_ind)
plt.step(sorted(Z), np.arange(1,n_ind+1)/float(n_ind),label = 'n = ' + str(n_ind))

plt.xlim([-3,3])
plt.xlabel('Observations')
plt.ylabel('Probability')
plt.title('Central Limit Theorem Verification - Part a')
plt.legend()

fig = plt.gcf()
fig.set_size_inches(10, 10)
plt.savefig('hw0_programming_plot_a.png')

# part b
plt.figure()
n = 40000
Z = np.random.randn(n_ind)
plt.step(sorted(Z), np.arange(1,n_ind+1)/float(n_ind),label='Gaussian')

k = [1,8,64,512]
# generate n of Y(k) random variables
n = 10000
for k_ind in k:
    Z = np.sum(np.sign(np.random.randn(n, k_ind))*np.sqrt(1./k_ind), axis=1)
    plt.step(sorted(Z), np.arange(1,n+1)/float(n),label=k_ind)

plt.xlim([-3,3])
plt.xlabel('Observations')
plt.ylabel('Probability')
plt.title('Central Limit Theorem Verification - Part b')
plt.legend()

fig = plt.gcf()
fig.set_size_inches(10, 10)
plt.savefig('hw0_programming_plot_b.png')
