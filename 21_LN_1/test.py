import numpy as np
import importlib


weyl = importlib.import_module("models.benchmark.weyl")

x =  np.arange(7*5*2)
y = np.arange(7)

x = x.reshape(7,5,2).astype(np.float32)
y = y.reshape(7,1).astype(np.float32)

t = weyl((x, y))
print(t)
