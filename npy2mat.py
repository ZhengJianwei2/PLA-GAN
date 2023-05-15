from re import X
from scipy import io 
import numpy as np

X=18
name1 = 'L506_{}_input'.format(X)
name2 = 'L506_{}_target'.format(X)
mat = np.load('test_img/{}.npy'.format(name1))
print(mat.max())
io.savemat('{}.mat'.format(name1),{'{}'.format(name1):mat})

mat2 = np.load('test_img/{}.npy'.format(name2))
print(mat2.max())
io.savemat('{}.mat'.format(name2),{'{}'.format(name2):mat2})
