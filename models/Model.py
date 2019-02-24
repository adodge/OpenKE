#coding:utf-8
import numpy as np
import tensorflow as tf

class Trainer(
    def __init__(cls,
            model,
            opt_method:str="SGD",
            alpha:float=0.001,
            model_params:dict=None):
        '''
        Create a TF graph using this model, including an optimizer, train op,
        saver, etc.
        '''
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            with sess.as_default():
                initializer = tf.contrib.layers.xavier_initializer(uniform = True)
                with tf.variable_scope("model", reuse=None, initializer = initializer):
                    trainModel = model(**model_params)
                    elif opt_method.lower() == "adagrad":
                        optimizer = tf.train.AdagradOptimizer(learning_rate = alpha, initial_accumulator_value=1e-20)
                    elif opt_method.lower() == "adadelta":
                        optimizer = tf.train.AdadeltaOptimizer(alpha)
                    elif opt_method.lower() == "adam":
                        optimizer = tf.train.AdamOptimizer(alpha)
                    elif opt_method.lower() == "sgd":
                        optimizer = tf.train.GradientDescentOptimizer(alpha)
                    else:
                        raise ValueError("Unknown opt_method: %s" % opt_method)
                    grads_and_vars = optimizer.compute_gradients(trainModel.loss)
                    train_op = optimizer.apply_gradients(grads_and_vars)
                saver = tf.train.Saver()
                sess.run(tf.initialize_all_variables())

        self.graph = graph
        self.sess = sess
        self.model = trainModel
        self.optimizer = optimizer
        self.grads_and_vars = grads_and_vars
        self.train_op = train_op
        self.saver = saver

    def train_step(self, batch_h, batch_t, batch_r, batch_y):
        feed_dict = {
            self.model.batch_h: batch_h,
            self.model.batch_t: batch_t,
            self.model.batch_r: batch_r,
            self.model.batch_y: batch_y
        }
        _, loss = self.sess.run([self.train_op, self.model.loss], feed_dict)
        return loss

    def run(self):
        with self.graph.as_default():
            with self.sess.as_default():
                for times in range(self.train_times):
                    res = 0.0
                    for batch in range(self.nbatches):
                        self.sampling()
                        res += self.train_step(self.batch_h, self.batch_t, self.batch_r, self.batch_y)
                    if self.log_on:
                        print(times)
                        print(res)
                    if self.exportName != None and (self.export_steps!=0 and times % self.export_steps == 0):
                        self.save_tensorflow()
                if self.exportName != None:
                    self.save_tensorflow()
                if self.out_path != None:
                    self.save_parameters(self.out_path)

    def save_tensorflow(self, exportName):
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.save(self.sess, exportName)

    def restore_tensorflow(self, importName):
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.restore(self.sess, importName)

class Model(object):
    '''
    A Model object represents a single instance of one of these embedding
    models.  When the object is initialized, it defines the shared parameters.
    It then exposes methods for constructing computation graphs using these
    parameters for training and testing.

    This lets us share parameters and do things like multi-objective training.

    Example ideal interactions:

    # Instantiates the parameters
    graph_embedding = TransE(n_entities=1000, n_relations=20)

    # Instantiates computation nodes
    loss = graph_embedding.loss(inputs,batch_size=5000)
    vectors = graph_embedding.embed(inputs)

    # Dump the parameters to a file
    # (Dumps both the model parameters and hyperparameters)
    graph_embedding.dump(filename)

    # Load a model from a file
    X = TransE.load(filename)
    '''

    def __init__(self,
            n_entities:int,
            n_relations:int,
            batch_size:int=5000,
            n_negative:int=1,

        # Number of entities in the training data (all entity ids should be
        # less than this number)
        self.n_entities = n_entities
        # Number of relations in the training data (all relation ids should be
        # less than this number)
        self.n_relations = n_relations
        # Training batch size XXX move to loss method
        self.batch_size = batch_size
        # Number of negative examples per positive example XXX move to loss method
        self.n_negative = n_negative

        # Allocate and define the model parameters
        with tf.name_scope("embedding"):
            self.embedding_def()

        # Construct TF graphs
        #with tf.name_scope("input"):
        #    self.input_def()

        #with tf.name_scope("predict"):
        #   self.predict_def()

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

        postive_h = tf.transpose(tf.reshape(batch_h[0:batch_size], [1, -1]), [1, 0])
        postive_t = tf.transpose(tf.reshape(batch_t[0:batch_size], [1, -1]), [1, 0])
        postive_r = tf.transpose(tf.reshape(batch_r[0:batch_size], [1, -1]), [1, 0])
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

    def embedding_def(self):
        raise NotImplementedError

    def predict_def(self):
        raise NotImplementedError
