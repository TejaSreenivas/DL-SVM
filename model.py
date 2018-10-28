import tensorflow as tf 
import numpy as np
import os

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
	
	def create_conv_layer(self,inp, filters, stride, padding):
		wt = tf.get_variable(name='weight',initializer=tf.truncated_normal(shape=filters,mean=0.0, stddev = 0.1))
		conv = tf.nn.conv2d(input=inp,filter=wt,strides=stride,padding=padding)
		b = tf.get_variable(name='bias',initializer=tf.zeros(shape=[filters[-1]]))
		out = tf.add(conv,b)
		act = tf.nn.relu(out)
		return act
	def cnn(self,data,is_train,prob_keep):
		DIM = 32
		CH = 3
		FLT = 3
		data = tf.layers.batch_normalization(data,training = is_train)
		with tf.variable_scope("conv_1",reuse = tf.AUTO_REUSE):
			conv1 = self.create_conv_layer(data,[FLT,FLT,CH,32],[1]*4,"SAME")
			conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
			conv1 = tf.layers.batch_normalization(conv1,training=is_train)
		DIM = int(DIM/2)
		CH = 32
		FLT = 3
		with tf.variable_scope('conv_2',reuse=tf.AUTO_REUSE):
			conv2 = self.create_conv_layer(conv1,[FLT,FLT,CH,64],[1]*4,"SAME")
			conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
			conv2 = tf.layers.batch_normalization(conv2,training=is_train)
		DIM = int(DIM/2)
		CH = 64
		FLT = 5
		
		with tf.variable_scope('conv_3',reuse=tf.AUTO_REUSE):
			conv3 = self.create_conv_layer(conv2,[FLT,FLT,CH,128],[1]*4,"SAME")
			conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
			conv3 = tf.layers.batch_normalization(conv3,training=is_train)
		
		"""
		DIM =int(DIM/2)
		CH = 128
		FLT = 5
		with tf.variable_scope('conv_4',reuse=tf.AUTO_REUSE):
			conv4 = self.create_conv_layer(conv3,[FLT,FLT,CH,256,[1]*4,"SAME"])
			conv4 = tf.nn.max_pool(conv4,[FLT,FLT,CH,256],[1]*4,"SAME")
			conv4 = tf.layers.batch_normalization(conv4,training=is_train)
		"""
		flat = tf.layers.flatten(conv3)

		h1 = tf.layers.dense(flat,units=1000)
		h1 = tf.nn.relu(h1)
		h1 = tf.layers.dropout(h1,rate = prob_keep, training = is_train)

		#logits
		#number of output classes + additional classes
		y = tf.layers.dense(h1,units=10+0)
		
		return y

	def __init__(self, data,is_train,prob_keep):
		self.logits = self.cnn(data,is_train,prob_keep)