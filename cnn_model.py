import tensorflow as tf
import numpy as np

class CNN:

	def total_params(self):
		count = 0
		info = ""
		for var in tf.trainable_variables():
			shape = var.get_shape()
			p = 1
			info = info + "varname : "+var.name+"-("
			print("varname : " + var.name,end=' ')
			l = []
			for d in shape:
				p*=d
				info = info + " " + str(d)
				l.append(d)
			count+=p
			print(l)
			info = info+" )$"
		info = info + "total param count : "+str(count)
		print("total number of trainable parameter : {}".format(count))
		return info

	def cnn(self, data, is_train, prob_keep):
		l1 = [32,32,3,64]
		conv1 = tf.layers.conv2d(data, filters=64, kernel_size=3,  padding='same', activation=tf.nn.relu, use_bias=True )
		conv1 = tf.layers.max_pooling2d(conv1, 2, 1, padding='same')
		conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3,  padding='same', activation=tf.nn.relu, use_bias=True )
		conv2 = tf.layers.max_pooling2d(conv2, 2, 1, padding='same')
		conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, use_bias=True )
		conv3 = tf.layers.max_pooling2d(conv3, 2, 1, padding='same')
		flat = tf.layers.flatten(conv3)
		h1 = tf.layers.dense(flat,units = 1000, activation = tf.nn.relu)
		h1 = tf.layers.dropout(h1, training = is_train)
		h1 = tf.layers.batch_normalization(h1, training = is_train)

		y = tf.layers.dense(h1, units = 10)
		return y
	def __init__(self, data, is_train, prob_keep):
		self.logits = self.cnn(data, is_train, prob_keep)

		
		