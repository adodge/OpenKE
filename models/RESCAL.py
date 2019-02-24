#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class RESCAL(Model):
    '''
    RESCAL is a tensor factorization approach to knowledge representation learning, 
    which is able to perform collective learning via the latent components of the factorization.
    '''

    def __init__(self,
            hidden_size:int=100,
            margin:float=1.0,
            **kwargs):

        self.hidden_size = hidden_size
        self.margin = margin

        super().__init__(**kwargs)


    def _calc(self, h, t, r):
        return h * tf.matmul(r, t)

    def embedding_def(self):
        '''
        Create variables for the model parameters
        '''
        self.ent_embeddings = tf.get_variable(
                name="ent_embeddings",
                shape=[self.n_entities, self.hidden_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform = False))

        self.rel_matrices = tf.get_variable(
                name="rel_matrices",
                shape=[self.n_relations, self.hidden_size * self.hidden_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform = False))

        self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
                                "rel_matrices":self.rel_matrices}
    def loss_batch(self, batch_h, batch_t, batch_r, batch_size, n_negative):

        inputs = self.split_inputs( batch_h, batch_t, batch_r, batch_size, n_negative)

        #To get positive triples and negative triples for training
        #The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
        #The shapes of neg_h, neg_t, neg_r are (batch_size, n_negative)
        pos_h, pos_t, pos_r = inputs['positive_h'],inputs['positive_t'],inputs['positive_r']
        neg_h, neg_t, neg_r = inputs['negative_h'],inputs['negative_t'],inputs['negative_r']

        #Embedding entities and relations of triples, e.g. p_h, p_t and p_r are embeddings for positive triples
        p_h = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_h), [-1, self.hidden_size, 1])
        p_t = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_t), [-1, self.hidden_size, 1])
        p_r = tf.reshape(tf.nn.embedding_lookup(self.rel_matrices, pos_r), [-1, self.hidden_size, self.hidden_size])
        n_h = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_h), [-1, self.hidden_size, 1])
        n_t = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_t), [-1, self.hidden_size, 1])
        n_r = tf.reshape(tf.nn.embedding_lookup(self.rel_matrices, neg_r), [-1, self.hidden_size, self.hidden_size])

        #The shape of _p_score is (batch_size, 1, hidden_size)
        #The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
        _p_score = tf.reshape(self._calc(p_h, p_t, p_r), [-1, 1, self.hidden_size])
        _n_score = tf.reshape(self._calc(n_h, n_t, n_r), [-1, self.n_negative, self.hidden_size])

        #The shape of p_score is (batch_size, 1)
        #The shape of n_score is (batch_size, 1)
        p_score =  tf.reduce_sum(tf.reduce_mean(_p_score, 1, keep_dims = False), 1, keep_dims = True)
        n_score =  tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims = False), 1, keep_dims = True)

        #Calculating loss to get what the framework will optimize
        return tf.reduce_sum(tf.maximum(n_score - p_score + self.margin, 0))
    
    def predict(self, predict_h, predict_t, predict_r):
        predict_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, predict_h), [-1, self.hidden_size, 1])
        predict_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, predict_t), [-1, self.hidden_size, 1])
        predict_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_matrices, predict_r), [-1, self.hidden_size, self.hidden_size])
        return -tf.reduce_sum(self._calc(predict_h_e, predict_t_e, predict_r_e), 1, keep_dims = False)
