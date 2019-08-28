import numpy as np
import time

a = np.ones(100000000)
b = np.ones(100000000)
c = np.ones(100000000)

start = time.time()

c = 2*a + b

end = time.time()

print(end - start)
