#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class DistMult(Model):
    '''
    DistMult is based on the bilinear model where each relation is represented by a diagonal rather than a full matrix. 
    DistMult enjoys the same scalable property as TransE and it achieves superior performance over TransE.
    '''

    def __init__(self,
            lmbda:float=0.0,
            hidden_size:int=100,
            **kwargs):

        self.lmbda = lmbda
        self.hidden_size = hidden_size

        super().__init__(**kwargs)

    
    def _calc(self, h, t, r):
        return h * r * t

    def embedding_def(self):
        '''
        Create variables for the model parameters
        '''

        self.ent_embeddings = tf.get_variable(
                name="ent_embeddings",
                shape=[self.n_entities, self.hidden_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform = True))

        self.rel_embeddings = tf.get_variable(
                name="rel_embeddings",
                shape=[self.n_relations, self.hidden_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform = True))

        self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
                                "rel_embeddings":self.rel_embeddings}

    def loss(self, h, t, r, y):
        #The shapes of h, t, r, y are (batch_size, 1 + n_negative)
        #Embedding entities and relations of triples
        e_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        e_t = tf.nn.embedding_lookup(self.ent_embeddings, t)
        e_r = tf.nn.embedding_lookup(self.rel_embeddings, r)

        #Calculating score functions for all positive triples and negative triples
        res = tf.reduce_sum(self._calc(e_h, e_t, e_r), 1, keep_dims = False)
        loss_func = tf.reduce_mean(tf.nn.softplus(- y * res))
        regul_func = tf.reduce_mean(e_h ** 2) + tf.reduce_mean(e_t ** 2) + tf.reduce_mean(e_r ** 2)

        #Calculating loss to get what the framework will optimize
        return loss_func + self.lmbda * regul_func

    def predict(self, predict_h, predict_t, predict_r):
        predict_h, predict_t, predict_r = self.get_predict_instance()
        predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_h)
        predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_t)
        predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, predict_r)
        return -tf.reduce_sum(self._calc(predict_h_e, predict_t_e, predict_r_e), 1, keep_dims = True)
