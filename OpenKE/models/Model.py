#coding:utf-8
import numpy as np
import tensorflow as tf
import pickle

class Model(object):
    '''
    A Model object represents a single instance of one of these embedding
    models.  When the object is initialized, it defines the shared parameters.
    It then exposes methods for constructing computation graphs using these
    parameters for training and testing.

    This lets us share parameters and do things like multi-objective training,
    or include this model as a part of a larger deep learning model.
    '''

    def __init__(self,
            n_entities:int,
            n_relations:int):

        # Number of entities in the training data (all entity ids should be
        # less than this number)
        self.n_entities = n_entities
        # Number of relations in the training data (all relation ids should be
        # less than this number)
        self.n_relations = n_relations

        self.args = ['n_entities', 'n_relations']
        self.parameter_lists = {}

        # Allocate and define the model parameters
        with tf.name_scope("embedding"):
            self.embedding_def()

    def split_inputs(self, batch_h, batch_t, batch_r, batch_size, n_negative):
        '''
        Input batches are passed in as three vectors of integers and a vector
        of floats.  These four vectors are aligned.

        The first _batch_size_ entries are positive examples and the remaining
        entries are negative examples.

        This takes the raw vectors as input and returns a dictionary of output
        nodes, with these splits applied.
        '''

        batch_seq_size = batch_size*(1+n_negative)

        positive_h = tf.transpose(tf.reshape(batch_h[0:batch_size], [1, -1]), [1, 0])
        positive_t = tf.transpose(tf.reshape(batch_t[0:batch_size], [1, -1]), [1, 0])
        positive_r = tf.transpose(tf.reshape(batch_r[0:batch_size], [1, -1]), [1, 0])
        negative_h = tf.transpose(tf.reshape(batch_h[batch_size:batch_seq_size], [n_negative, -1]), perm=[1, 0])
        negative_t = tf.transpose(tf.reshape(batch_t[batch_size:batch_seq_size], [n_negative, -1]), perm=[1, 0])
        negative_r = tf.transpose(tf.reshape(batch_r[batch_size:batch_seq_size], [n_negative, -1]), perm=[1, 0])

        return {
            'positive_h': positive_h,
            'positive_t': positive_t,
            'positive_r': positive_r,
            'negative_h': negative_h,
            'negative_t': negative_t,
            'negative_r': negative_r,
        }

    def loss(self, batch_h, batch_t, batch_r, batch_y, batch_size, n_negative):
        '''
        There are two types of loss functions defined in this module.

        Analogy, ComplEx, and DistMult take arguments (h,t,r,y).  The rest take
        (h,t,r,batch_size,n_negative).

        y is an aligned vector of target similarities, whereas batch_size and
        n_negative allow the loss function to determine which (h,t,r) examples
        are positive or negative (using the split_inputs method).

        The data loader / sampling logic generates both the batch_y and
        (batch_size, n_negative) values.  This function takes the union of
        these arguments are delegates to the appropriate loss function for the
        model.
        '''

        if hasattr(self, 'loss_y'):
            return self.loss_y(batch_h, batch_t, batch_r, batch_y)
        if hasattr(self, 'loss_batch'):
            return self.loss_batch(batch_h, batch_t, batch_r, batch_size, n_negative)
        raise NotImplementedError("Model doesn't seem to have a loss function defined.")

    def predict(self, predict_h, predict_t, predict_r):
        raise NotImplementedError

    @property
    def arguments(self):
        '''
        Return a dictionary of the keyword arguments used to create this model.
        '''
        out = {}
        for key in self.args:
            out[key] = getattr(self,key)
        return out

    @property
    def parameters(self):
        '''
        Return the parameter values as a dictionary of numpy arrays
        '''
        out = {}
        for key,node in self.parameter_lists.items():
            out[key] = tf.get_default_session().run(node)
        return out

    @parameters.setter
    def parameters(self, params):
        for key,value in params.items():
            self.parameter_lists[key].assign(value).eval()

    def dump(self, fp):
        with open(fp, 'wb') as fd:
            pickle.dump(self.__class__, fd)
            pickle.dump(self.arguments, fd)
            pickle.dump(self.parameters, fd)

    @classmethod
    def load(cls, fp):
        with open(fp, 'rb') as fd:
            modelcls = pickle.load(fd)
            args = pickle.load(fd)
            params = pickle.load(fd)

        model = modelcls(**args)
        model.parameters = params
        return model
