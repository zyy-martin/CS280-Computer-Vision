import numpy as np
import cPickle
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from learner import *
from dataset import Dataset,SingleDataset
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate',type = float,default = 0.001) 
	parser.add_argument('--batch_size',type = int,default = 100) 
	parser.add_argument('--num_epoches',type = int, default = 10)
	parser.add_argument('--task',type = str, 
		default = 'finetune')
	parser.add_argument('--dataset',type = str, default = 'mnist')  
	parser.add_argument('--train',
						type = str,
						default = 'train') 
	parser.add_argument('--saved_model',
						type = str,
						default = 'trained_model.ckpt') 

	args = parser.parse_args()   

	if args.dataset == 'mnist': 
		num_classes = 10
		img_height = img_width = 28
		dataset = input_data.read_data_sets('MNIST_data', 
			one_hot=False,reshape = False) 

	if args.dataset == 'places':
		num_classes = 100
		img_height = img_width = 128
		# with open('data/dataset.save','rb') as f:
		# 	dataset = cPickle.load(f)
		print 'Read in dataset'
		dataset = Dataset()

	model = Learner(dataset,
		learning_rate = args.learning_rate,
		num_classes = num_classes,
		batch_size = args.batch_size,
		img_height = img_height,
		img_width = img_width,
		num_epoches = args.num_epoches,
		task=args.task,
		saved_model = 'saved_model') 

	if args.train == 'train':
		model.train()
	elif args.train == 'eval':
		matrix = model.eval()
		np.savetxt('output.txt', matrix)
		print 'output saved'
	elif args.train == 'val':
		model.validate()

if __name__ == '__main__':
	main()