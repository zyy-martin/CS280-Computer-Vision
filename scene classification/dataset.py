import numpy as np
import cPickle
class Dataset(object): 
	def __init__(self):
		train_path = 'data/rlt_train.npy'
		val_path = 'data/rlt_val.npy'
		test_path = 'data/rlt_test.npy'
		
		self.train = SingleDataset(train_path)
		print 'read in training data'
		self.validation = SingleDataset(val_path)
		print 'read in validation data'
		
		self.test = SingleDataset(test_path) 
		print 'read in test data'

class SingleDataset:
	def __init__(self, path): 

		obj = np.load(path).item()
		X = obj['X']
		filenames = obj['filenames']

		if path.endswith('_test.npy'): 
			y = np.zeros(len(filenames),dtype = int) 
		else:
			y = obj['y']

		self.num_examples = len(y) 
		self.images = np.array([X[filename] for filename in filenames])   
		self.labels = np.array(y)

		permutation = np.random.permutation(self.num_examples)
		self.images = self.images[permutation]
		self.labels = self.labels[permutation]       

		# # small data experiments.
		# self.num_examples = 1000
		# self.images = self.images[:self.num_examples]
		# self.labels = self.labels[:self.num_examples]

		self.batch_id = 0
		
	def next_batch(self, batch_size):
		""" Return a batch of data. When dataset end is reached, start over.
		"""
		if self.batch_id == self.num_examples:
			self.batch_id = 0 
			permutation = np.random.permutation(self.num_examples)
			self.images = self.images[permutation]
			self.labels = self.labels[permutation]

		batch_data = (self.images[self.batch_id:min(self.batch_id +
										  batch_size, self.num_examples)]) 

		batch_labels = (self.labels[self.batch_id:min(self.batch_id +
										  batch_size, self.num_examples)]) 
		self.batch_id = min(self.batch_id + batch_size, len(self.images))
		return batch_data, batch_labels
 
# def main():
#     dataset = Dataset()
#     with open('dataset.save','wb') as f:
#         cPickle.dump(dataset,f)

# if __name__ == '__main__':
#     main()