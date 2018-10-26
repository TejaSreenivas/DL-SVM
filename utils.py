import tensorflow as tf 
import numpy as np

def oneHot(data,cls):
	x = data.reshape(-1)
	oh = np.eye(cls)[x].astype(np.float32)
	return oh