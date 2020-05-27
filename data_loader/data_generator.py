import numpy as np
import h5py

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

class DataGenerator:
    def __init__(self, config):
        self.config = config

        # load data here
        train_dataset = h5py.File('datasets/train_signs.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
        
        test_dataset = h5py.File('datasets/test_signs.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        # classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        self.X_train = train_set_x_orig / 255.
        self.Y_train = convert_to_one_hot(train_set_y_orig, 6).T

        self.X_test = test_set_x_orig / 255.
        self.Y_test = convert_to_one_hot(test_set_y_orig, 6).T

        self.nSamples = self.X_train.shape[0]  # number of training examples

        print("shape X", self.X_train.shape)
        print("shape Y", self.Y_train.shape)
        print("number of samples", self.nSamples)
        
        # Shuffle (X, Y)
        permutation = list(np.random.permutation(self.nSamples))
        self.shuffled_X = self.X_train[permutation,:,:,:]
        self.shuffled_Y = self.Y_train[permutation,:]

        # minibatch counter
        self.minibatch_idx = 0
        
    def next_batch(self, batch_size):
        
        idx = np.array(
            range(self.minibatch_idx, self.minibatch_idx + batch_size)
        ) % self.nSamples  # circular

        self.minibatch_idx = ( self.minibatch_idx + batch_size ) % self.nSamples  # update last index

        yield self.shuffled_X[idx], self.shuffled_Y[idx]
