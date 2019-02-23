#coding:utf-8
import numpy as np
import tensorflow as tf

class Model(object):
	def __init__(self,
            n_entities:int,
            n_relations:int,
            batch_size:int=5000,
            negative_ent:int=1,
            negative_rel:int=0):

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.batch_size = batch_size
        self.negative_ent = negative_ent
        self.negative_rel = negative_rel

        # Construct TF graphs
		with tf.name_scope("input"):
			self.input_def()

		with tf.name_scope("embedding"):
			self.embedding_def()

		with tf.name_scope("loss"):
			self.loss_def()

		with tf.name_scope("predict"):
			self.predict_def()

    @property
    def batch_seq_size(self):
        return self.batch_size*(1+self.negative_ent+self.negative_rel)

	def get_positive_instance(self, in_batch = True):
		if in_batch:
			return [self.postive_h, self.postive_t, self.postive_r]
		else:
			return [self.batch_h[0:self.batch_size], \
			self.batch_t[0:self.batch_size], \
			self.batch_r[0:self.batch_size]]

	def get_negative_instance(self, in_batch = True):
		if in_batch:
			return [self.negative_h, self.negative_t, self.negative_r]
		else:
			return [self.batch_h[self.batch_size:self.batch_seq_size],\
			self.batch_t[self.batch_size:self.batch_seq_size],\
			self.batch_r[self.batch_size:self.batch_seq_size]]

	def get_all_instance(self, in_batch = False):
		if in_batch:
			return [tf.transpose(tf.reshape(self.batch_h, [1 + self.negative_ent + self.negative_rel, -1]), [1, 0]),\
			tf.transpose(tf.reshape(self.batch_t, [1 + self.negative_ent + self.negative_rel, -1]), [1, 0]),\
			tf.transpose(tf.reshape(self.batch_r, [1 + self.negative_ent + self.negative_rel, -1]), [1, 0])]
		else:
			return [self.batch_h, self.batch_t, self.batch_r]

	def get_all_labels(self, in_batch = False):
		if in_batch:
			return tf.transpose(tf.reshape(self.batch_y, [1 + self.negative_ent + self.negative_rel, -1]), [1, 0])
		else:
			return self.batch_y

	def get_predict_instance(self):
		return [self.predict_h, self.predict_t, self.predict_r]

	def input_def(self):
		self.batch_h = tf.placeholder(tf.int64, [self.batch_seq_size])
		self.batch_t = tf.placeholder(tf.int64, [self.batch_seq_size])
		self.batch_r = tf.placeholder(tf.int64, [self.batch_seq_size])
		self.batch_y = tf.placeholder(tf.float32, [self.batch_seq_size])
		self.postive_h = tf.transpose(tf.reshape(self.batch_h[0:self.batch_size], [1, -1]), [1, 0])
		self.postive_t = tf.transpose(tf.reshape(self.batch_t[0:self.batch_size], [1, -1]), [1, 0])
		self.postive_r = tf.transpose(tf.reshape(self.batch_r[0:self.batch_size], [1, -1]), [1, 0])
		self.negative_h = tf.transpose(tf.reshape(self.batch_h[self.batch_size:self.batch_seq_size], [self.negative_ent + self.negative_rel, -1]), perm=[1, 0])
		self.negative_t = tf.transpose(tf.reshape(self.batch_t[self.batch_size:self.batch_seq_size], [self.negative_ent + self.negative_rel, -1]), perm=[1, 0])
		self.negative_r = tf.transpose(tf.reshape(self.batch_r[self.batch_size:self.batch_seq_size], [self.negative_ent + self.negative_rel, -1]), perm=[1, 0])
		self.predict_h = tf.placeholder(tf.int64, [None])
		self.predict_t = tf.placeholder(tf.int64, [None])
		self.predict_r = tf.placeholder(tf.int64, [None])
		self.parameter_lists = []

	def embedding_def(self):
        raise NotImplementedError

	def loss_def(self):
        raise NotImplementedError

	def predict_def(self):
        raise NotImplementedError
