#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class TransH(Model):
    '''
    To preserve the mapping propertities of 1-N/N-1/N-N relations, 
    TransH inperprets a relation as a translating operation on a hyperplane. 
    '''

    def __init__(self,
            rel_size:int=100,
            hidden_size:int=100,
            margin:float=1.0,
            **kwargs):

        self.rel_size = rel_size
        self.hidden_size = hidden_size
        self.margin = margin

        super().__init__(**kwargs)
        self.args.extend(['margin', 'hidden_size', 'rel_size'])

    def _transfer(self, e, n):
        return e - tf.reduce_sum(e * n, 1, keepdims = True) * n

    def _calc(self, h, t, r):
        return abs(h + r - t)

    def embedding_def(self):
        '''
        Create variables for the model parameters
        '''
        self.ent_embeddings = tf.get_variable(
                name = "ent_embeddings",
                shape = [self.n_entities, self.hidden_size],
                initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        self.rel_embeddings = tf.get_variable(
                name = "rel_embeddings",
                shape = [self.n_relations, self.hidden_size],
                initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        self.normal_vectors = tf.get_variable(
                name = "normal_vectors",
                shape = [self.n_entities, self.hidden_size],
                initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        self.parameter_lists.update({"ent_embeddings":self.ent_embeddings, \
                                "rel_embeddings":self.rel_embeddings, \
                                "normal_vectors":self.normal_vectors})

    def loss_batch(self, batch_h, batch_t, batch_r, batch_size, n_negative):

        pos_h = batch_h[:batch_size]
        pos_t = batch_t[:batch_size]
        pos_r = batch_r[:batch_size]
        neg_h = batch_h[batch_size:]
        neg_t = batch_t[batch_size:]
        neg_r = batch_r[batch_size:]

        #Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
        pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
        pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
        pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
        neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
        neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
        neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)

        #Getting the required normal vectors of planes to transfer entity embeddings
        pos_norm = tf.nn.embedding_lookup(self.normal_vectors, pos_r)
        neg_norm = tf.nn.embedding_lookup(self.normal_vectors, neg_r)
        
        pos_h_e = tf.nn.l2_normalize(pos_h_e, 1)
        pos_t_e = tf.nn.l2_normalize(pos_t_e, 1)
        pos_r_e = tf.nn.l2_normalize(pos_r_e, 1)
        neg_h_e = tf.nn.l2_normalize(neg_h_e, 1)
        neg_t_e = tf.nn.l2_normalize(neg_t_e, 1)
        neg_r_e = tf.nn.l2_normalize(neg_r_e, 1)
        
        pos_norm = tf.nn.l2_normalize(pos_norm, 1)
        neg_norm = tf.nn.l2_normalize(neg_norm, 1)
    
        #Calculating score functions for all positive triples and negative triples
        p_h = self._transfer(pos_h_e, pos_norm)
        p_t = self._transfer(pos_t_e, pos_norm)
        p_r = pos_r_e
        n_h = self._transfer(neg_h_e, neg_norm)
        n_t = self._transfer(neg_t_e, neg_norm)
        n_r = neg_r_e

        #Calculating score functions for all positive triples and negative triples
        #The shape of _p_score is (1, batch_size, hidden_size)
        #The shape of _n_score is (n_negative, batch_size, hidden_size)
        _p_score = self._calc(p_h, p_t, p_r)
        _p_score = tf.reshape(_p_score, [1, -1, self.rel_size])
        _n_score = self._calc(n_h, n_t, n_r)
        _n_score = tf.reshape(_n_score, [n_negative, -1, self.rel_size])

        #The shape of p_score is (batch_size, 1)
        #The shape of n_score is (batch_size, 1)
        p_score =  tf.reduce_sum(tf.reduce_mean(_p_score, 0, keepdims = False), 1, keepdims = True)
        n_score =  tf.reduce_sum(tf.reduce_mean(_n_score, 0, keepdims = False), 1, keepdims = True)

        #Calculating loss to get what the framework will optimize
        return tf.reduce_sum(tf.maximum(p_score - n_score + self.margin, 0))

    def predict(self, predict_h, predict_t, predict_r):
        predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_h)
        predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_t)
        predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, predict_r)
        predict_norm = tf.nn.embedding_lookup(self.normal_vectors, predict_r)
        
        predict_h_e = tf.nn.l2_normalize(predict_h_e, 1)
        predict_t_e = tf.nn.l2_normalize(predict_t_e, 1)
        predict_r_e = tf.nn.l2_normalize(predict_r_e, 1)
        predict_norm = tf.nn.l2_normalize(predict_norm, 1)

        h_e = self._transfer(predict_h_e, predict_norm)
        t_e = self._transfer(predict_t_e, predict_norm)
        r_e = predict_r_e
        return tf.reduce_sum(self._calc(h_e, t_e, r_e), 1, keepdims = True)
