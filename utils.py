import tensorflow as tf 
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

#scaling the data
def rescaling(data):
	scaler = MinMaxScaler(feature_range=(0,1))
	rescaled = scaler.fit_transform(data)
	return rescaled

# normalization
def 