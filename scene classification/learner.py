from network import shallow_resnet
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist

import cPickle
import numpy as np 

class Learner(object):
	def __init__(self,dataset,
		learning_rate = 0.001,
		num_classes = 10,
		batch_size = 50,
		img_height = 32,
		img_width = 32,
		num_epoches = 20,
		task='finetune',
		saved_model = 'trained_model.ckpt'):

		self.dataset = dataset
		self.learning_rate = learning_rate
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.img_height = img_height
		self.img_width = img_width
		self.num_epoches = num_epoches
		self.task = task
		self.saved_model = saved_model

		self.build_graph()

	def build_graph(self):
		# Put placeholders
		self.images = tf.placeholder(tf.float32,
			[self.batch_size, self.img_height, self.img_width, 3],name = 'images')

		# images = tf.random_crop(self.images, 
		# 	[self.batch_size, 64, 64, 3])

		# images = tf.image.resize_images(images, [128,128])
		# Randomly flip the image horizontally.
		# images = tf.image.random_flip_left_right(images)

		# # Because these operations are not commutative, consider randomizing
		# # the order their operation.
		# images = tf.image.random_brightness(images, max_delta=63)
		# images = tf.image.random_contrast(images, lower=0.2, upper=1.8)

		self.labels = tf.placeholder(tf.int64,[self.batch_size],name = 'labels') 
		one_hot_labels = tf.one_hot(self.labels,self.num_classes)

		self.train_phase = tf.placeholder(tf.bool, name='train_phase') 

		# Lay down graph for the loss. 
		# logits = shallow_resnet(images,
		# 	self.num_classes,
		# 	is_training = self.train_phase)
		logits = shallow_resnet(self.images,
			self.num_classes,
			is_training = self.train_phase)
		self.logits = tf.reshape(logits,[self.batch_size,self.num_classes])

		self.loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,
			 logits=self.logits))

		self.predictions = tf.argmax(self.logits,1)
		correct_predictions = tf.equal(self.labels,
			self.predictions)

		self.top5 = tf.nn.top_k(self.logits,k=5, sorted = True)

		self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))

		self.optimum = \
		tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		# Lay down computational graph for network. 
		if self.task == 'finetune':
			# Get list of variables to restore  
			variables_to_restore = []
			exclusions = ["resnet_v1_50/logits","biases",
			"resnet_v1_50/block3","resnet_v1_50/block4"]
			for var in slim.get_model_variables(): 
				excluded = False
				for exclusion in exclusions: 
					if var.op.name.startswith(exclusion) or \
					var.op.name.endswith(exclusion):
						excluded = True
						break
				 
				if not excluded:
					variables_to_restore.append(var)

		elif str.startswith(self.task,'continue_training'): 
			variables_to_restore = slim.get_model_variables()
		elif str.startswith(self.task,'validation'):
			variables_to_restore = slim.get_model_variables()
		else:
			variables_to_restore = slim.get_model_variables()
			

		self.restorer = tf.train.Saver(variables_to_restore) 
		self.saver = tf.train.Saver()


	def train(self):   
		with tf.Session() as sess:
			init_op = tf.global_variables_initializer()
			sess.run(init_op) 

			# Restore variables from disk.
			if self.task == 'finetune':
				self.restorer.restore(sess, "./resnet_v1_50.ckpt")
				print("Model restored.") 
			elif str.startswith(self.task,'continue_training'):
				print('haha')
				model_tobe_restored = './saved_model-100.ckpt'

				self.restorer.restore(sess, model_tobe_restored)
			elif str.startswith(self.task,'validation'):
				model_tobe_restored = self.task.split(':')[1]
				self.restorer.restore(sess, model_tobe_restored)

			n_samples = self.dataset.train.num_examples
			num_batches = int(n_samples / self.batch_size)
			n_val_samples = self.dataset.validation.num_examples
			num_val_batches = \
			int(n_val_samples / self.batch_size)

			# Training
			print 'The training stage...'
			for epoch in xrange(self.num_epoches):
				avg_loss_value = 0.
				avg_train_acc = 0.0

				for b in xrange(num_batches):
					batch_images,batch_labels = self.dataset.train.next_batch(batch_size=self.batch_size)

					if batch_images.shape[-1] == 1:
						batch_images = np.tile(batch_images,(1,1,1,3))

					_,loss_val,train_acc = sess.run([self.optimum,
						self.loss,self.accuracy],
						feed_dict = {
						self.images: batch_images,
						self.labels: batch_labels,
						self.train_phase: True}) 

					avg_loss_value += loss_val / n_samples * self.batch_size
					avg_train_acc += train_acc / n_samples * self.batch_size

					print loss_val, train_acc

					current_ratio = float(n_samples)/(b * self.batch_size + self.batch_size)

					with open("training_log.txt", "a") as f:
						f.write(str(avg_loss_value * current_ratio)+','+
							str(avg_train_acc * current_ratio) +'\n')
				 
				avg_val_acc = 0.0

				for b in xrange(num_val_batches):

					val_images,val_labels = self.dataset.validation.next_batch(self.batch_size)

					if val_images.shape[-1] == 1:
						val_images = np.tile(val_images,(1,1,1,3))

					val_acc = sess.run(self.accuracy,
						feed_dict = {self.images:val_images, 
						self.labels: val_labels,
						self.train_phase: False})

					print val_acc

					avg_val_acc += val_acc /  n_val_samples * self.batch_size
	

				if (epoch+1) % 10 == 0:
					self.saver.save(sess,
						self.saved_model + "-{}.ckpt".format(epoch+1))

				log = "Epoch: %s Training Loss: %s Train Accuracy: %s Val Accuracy: %s"%(epoch,avg_loss_value,
					avg_train_acc,avg_val_acc)

				with open("epoch_log.txt", "a") as f:
					f.write(log+'\n')
					
				print log
			   

			# Save the model.
			saver = tf.train.Saver()
			saver.save(sess, self.saved_model) 


	def eval(self): 
		with tf.Session() as sess:
			init_op = tf.global_variables_initializer()
			sess.run(init_op) 
			# model_tobe_restored = self.task.split(':')[1]
			self.restorer.restore(sess,"./saved_model-10.ckpt")
			print 'hehe'
			n_samples = self.dataset.test.num_examples
			num_batches = int(n_samples / self.batch_size) 

			# Test
			output = np.zeros((n_samples,5),dtype = int)
			print 'The test stage...'  
			for b in xrange(num_batches):

				test_images,_ = self.dataset.test.next_batch(self.batch_size)

				if test_images.shape[-1] == 1:
					test_images = np.tile(test_images,(1,1,1,3))

				test_values, test_labels = sess.run(self.top5,
					feed_dict = {self.images:test_images, 
					self.train_phase: False})
				print test_labels
				output[b*self.batch_size: (b+1)*self.batch_size]=\
				test_labels

		return output

	def validate(self):
		with tf.Session() as sess: 
			init_op = tf.global_variables_initializer()
			sess.run(init_op) 
			model_tobe_restored = self.task.split(':')[1]
			self.restorer.restore(sess, "./saved_model-100.ckpt") 
			avg_val_acc = 0.0
			n_val_samples = self.dataset.validation.num_examples
			num_val_batches = int(n_val_samples / self.batch_size)

			for b in xrange(num_val_batches):

				val_images,val_labels = self.dataset.validation.next_batch(self.batch_size)

				if val_images.shape[-1] == 1:
					val_images = np.tile(val_images,(1,1,1,3))

				val_acc = sess.run(self.accuracy,
					feed_dict = {self.images:val_images, 
					self.labels: val_labels,
					self.train_phase: False})

				print val_acc

				avg_val_acc += val_acc /  n_val_samples * self.batch_size

			print 'Val Accuracy: {}'.format(avg_val_acc)
			


















		
