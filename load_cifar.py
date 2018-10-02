import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def unpickle(file):
	with open(file,'rb') as f:
		data = pickle.load(f, encoding='latin-1')
		return data

def load_data(path='C:/Users/k_tej/Documents/TEJA/ML_resources/ML_concepts/CNN/data/cifar-10-python.tar/cifar-10-python/cifar-10-batches-py'):

	train_data = None
	train_labels = []

	for i in range(1,6):
		data_dict = unpickle(path+'/data_batch_{}'.format(i))
		if i == 1:
			train_data = data_dict['data']
		else:
			train_data = np.vstack((train_data,data_dict['data']))
		train_labels+=data_dict['labels']

	#reshaping train data
	train_data = train_data.reshape(len(train_data),3,32,32)
	train_data = np.transpose(train_data,[0,2,3,1])

	train_labels = np.array(train_labels)

	# for test data
	test_data = None

	data_dict = unpickle(path+'/test_batch')
	test_data = data_dict['data'].reshape(len(data_dict['data']),3,32,32)
	test_data = np.transpose(test_data,[0,2,3,1])
	test_labels = np.array(data_dict['labels'])
	data = dict()
	data['train_x'] = train_data
	data['train_y'] = train_labels
	data['test_x'] = test_data
	data['test_y'] = test_labels
	return data

if __name__ == "__main__":
	tr_x,tr_y,ts_x,ts_y = load_data()
	print(tr_x.shape)
	print(ts_x.shape)
	plt.imshow(tr_x[5])
	plt.show()