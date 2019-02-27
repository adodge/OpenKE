#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class TransR(Model):
    '''
    TransR first projects entities from entity space to corresponding relation space 
    and then builds translations between projected entities. 
    '''

    def __init__(self,
            rel_size:int=100,
            ent_size:int=100,
            hidden_size:int=100,
            margin:float=1.0,
            **kwargs):

        self.rel_size = rel_size
        self.ent_size = ent_size
        self.hidden_size = hidden_size
        self.margin = margin

        super().__init__(**kwargs)
        self.args.extend(['margin', 'hidden_size', 'rel_size', 'ent_size'])

    def _transfer(self, transfer_matrix, embeddings):
        return tf.batch_matmul(transfer_matrix, embeddings)

    def _calc(self, h, t, r):
        return abs(h + r - t)

    def embedding_def(self):
        '''
        Create variables for the model parameters
        '''
        self.ent_embeddings = tf.get_variable(
                name = "ent_embeddings",
                shape = [self.n_entities, self.ent_size],
                initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        self.rel_embeddings = tf.get_variable(
                name = "rel_embeddings",
                shape = [self.n_relations, self.rel_size],
                initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        self.transfer_matrix = tf.get_variable(
                name = "transfer_matrix",
                shape = [self.n_relations, self.ent_size * self.rel_size],
                initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        self.parameter_lists.update({"ent_embeddings":self.ent_embeddings, \
                                "rel_embeddings":self.rel_embeddings, \
                                "transfer_matrix":self.transfer_matrix})

    def loss_batch(self, batch_h, batch_t, batch_r, batch_size, n_negative):

        inputs = self.split_inputs( batch_h, batch_t, batch_r, batch_size,
                n_negative)

        #To get positive triples and negative triples for training
        #The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
        #The shapes of neg_h, neg_t, neg_r are (batch_size, n_negative)
        pos_h, pos_t, pos_r = inputs['positive_h'],inputs['positive_t'],inputs['positive_r']
        neg_h, neg_t, neg_r = inputs['negative_h'],inputs['negative_t'],inputs['negative_r']

        #Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
        pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_h), [-1, self.ent_size, 1])
        pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_t), [-1, self.ent_size, 1])
        pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, pos_r), [-1, self.rel_size])
        neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_h), [-1, self.ent_size, 1])
        neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_t), [-1, self.ent_size, 1])
        neg_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, neg_r), [-1, self.rel_size])

        #Getting the required mapping matrices
        pos_matrix = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, pos_r), [-1, self.rel_size, self.ent_size])
        neg_matrix = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, neg_r), [-1, self.rel_size, self.ent_size])

        #Calculating score functions for all positive triples and negative triples
        p_h = tf.reshape(self._transfer(pos_matrix, pos_h_e), [-1, self.rel_size])
        p_t = tf.reshape(self._transfer(pos_matrix, pos_t_e), [-1, self.rel_size])
        p_r = pos_r_e
        n_h = tf.reshape(self._transfer(neg_matrix, neg_h_e), [-1, self.rel_size])
        n_t = tf.reshape(self._transfer(neg_matrix, neg_t_e), [-1, self.rel_size])
        n_r = neg_r_e

        #The shape of _p_score is (batch_size, 1, hidden_size)
        #The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
        _p_score = self._calc(p_h, p_t, p_r)
        _p_score = tf.reshape(_p_score, [-1, 1, self.rel_size])
        _n_score = self._calc(n_h, n_t, n_r)
        _n_score = tf.reshape(_n_score, [-1, self.n_negative, self.rel_size])

        #The shape of p_score is (batch_size, 1)
        #The shape of n_score is (batch_size, 1)
        p_score =  tf.reduce_sum(tf.reduce_mean(_p_score, 1, keepdims = False), 1, keepdims = True)
        n_score =  tf.reduce_sum(tf.reduce_mean(_n_score, 1, keepdims = False), 1, keepdims = True)

        #Calculating loss to get what the framework will optimize
        return tf.reduce_sum(tf.maximum(p_score - n_score + self.margin, 0))

    def predict(self, predict_h, predict_t, predict_r):
        predict_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, predict_h), [-1, self.ent_size, 1])
        predict_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, predict_t), [-1, self.ent_size, 1])
        predict_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, predict_r), [-1, self.rel_size])
        predict_matrix = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, predict_r), [-1, self.rel_size, self.ent_size])
        h_e = tf.reshape(self._transfer(predict_matrix, predict_h_e), [-1, self.rel_size])
        t_e = tf.reshape(self._transfer(predict_matrix, predict_t_e), [-1, self.rel_size])
        r_e = predict_r_e
        return tf.reduce_sum(self._calc(h_e, t_e, r_e), 1, keepdims = True)
