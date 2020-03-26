import numpy as np
test = np.ones([3,3])
test[1][2] =0
print(test)
print(test.reshape((-1)))
