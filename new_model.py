import tensorflow as tf 

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

	def create_conv_layer(self, inp, filters, stride, padding):
		wt =  tf.get_variable(name='weight', initializer=tf.truncated_normal(shape=filters, mean=0.0, stddev = 1.0))
		conv = tf.nn.conv2d(input = inp, filter = wt, strides = stride, padding = padding)
		b = tf.get_variable(name='bias', initializer = tf.zeros(shape=[filters[-1]]))
		out = tf.add(conv, b)
		return out

	def cnn(self, data, is_train, prob_keep):
		l1 = [3,3,3,64]
		with tf.variable_scope("conv_1", reuse=tf.AUTO_REUSE):
			conv1 = self.create_conv_layer(data, l1, [1]*4, "SAME")
			conv1 = tf.layers.batch_normalization(conv1, training = is_train)
			conv1 = tf.nn.relu(conv1)
			conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME")
		l2 = [5,5,l1[-1],128]
		with tf.variable_scope("conv_2", reuse=tf.AUTO_REUSE):
			conv2 = self.create_conv_layer(conv1, l2, [1]*4, "SAME")
			conv2 = tf.layers.batch_normalization(conv2, training = is_train)
			conv2 = tf.nn.relu(conv2)
			conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME")
		l3 = [5,5,l2[-1],256]
		with tf.variable_scope("conv_3", reuse=tf.AUTO_REUSE):
			conv3 = self.create_conv_layer(conv2, l3, [1]*4, "SAME")
			conv3 = tf.layers.batch_normalization(conv3, training = is_train)
			conv3 = tf.nn.relu(conv3)
			conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME")
		
		flat = tf.layers.flatten(conv3)

		h1 = tf.layers.dense(flat, units = 1000)
		h1 = tf.layers.batch_normalization(h1, training = is_train)
		h1 = tf.nn.relu(h1)
		h1 = tf.layers.dropout(h1, rate = prob_keep, training = is_train)

		y = tf.layers.dense(h1, units = 10)
		return y

	def __init__(self, data, is_train, prob_keep):
		self.logits = self.cnn(data, is_train, prob_keep)
		