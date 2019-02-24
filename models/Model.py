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
    A Model is the chunk of a TF graph that implements one of the graph
    embedding methods.  The training, saving, etc, take place in the
    Trainer class.

    Separating these concerns is, I think, a tensorflow idiom to make it easier
    to combine multiple models into one computation graph to optimize it all
    together.

    Normally, I think we would want to arrange it so that we initialize the
    model and all its parameters when we __init__ this class.  Then we have
    methods on that instance to actually create bits of computation graph, with
    inputs passed in and returning the outputs.

    This lets us share parameters and do things like multi-objective training.

    # Instantiates the parameters
    graph_embedding = TransE(n_entities=1000, n_relations=20)

    # Instantiates computation nodes
    loss = graph_embedding.loss(inputs,batch_size=5000)
    vectors = graph_embedding.embed(inputs)

    Note this is not how it is in the base project, so some reworking is
    needed.
    '''
	def __init__(self,
            n_entities:int,
            n_relations:int,
            batch_size:int=5000,
            n_negative:int=1,

        # Number of entities in the training data
        self.n_entities = n_entities
        # Number of relations in the training data
        self.n_relations = n_relations
        # Training batch size
        self.batch_size = batch_size
        # Number of negative examples per positive example
        self.n_negative = n_negative

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
        return self.batch_size*(1+self.n_negative)

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
			return [tf.transpose(tf.reshape(self.batch_h, [1 + self.n_negative, -1]), [1, 0]),\
			tf.transpose(tf.reshape(self.batch_t, [1 + self.n_negative, -1]), [1, 0]),\
			tf.transpose(tf.reshape(self.batch_r, [1 + self.n_negative, -1]), [1, 0])]
		else:
			return [self.batch_h, self.batch_t, self.batch_r]

	def get_all_labels(self, in_batch = False):
		if in_batch:
			return tf.transpose(tf.reshape(self.batch_y, [1 + self.n_negative, -1]), [1, 0])
		else:
			return self.batch_y

	def get_predict_instance(self):
		return [self.predict_h, self.predict_t, self.predict_r]

	def input_def(self):
        '''
        Input batches are passed in as three lists of integers and a list of
        floats.  These four lists are aligned.

        The first _batch_size_ entries are positive examples and the remaining
        entries are negative examples.
        '''
		self.batch_h = tf.placeholder(tf.int64, [self.batch_seq_size])
		self.batch_t = tf.placeholder(tf.int64, [self.batch_seq_size])
		self.batch_r = tf.placeholder(tf.int64, [self.batch_seq_size])
		self.batch_y = tf.placeholder(tf.float32, [self.batch_seq_size])
		self.postive_h = tf.transpose(tf.reshape(self.batch_h[0:self.batch_size], [1, -1]), [1, 0])
		self.postive_t = tf.transpose(tf.reshape(self.batch_t[0:self.batch_size], [1, -1]), [1, 0])
		self.postive_r = tf.transpose(tf.reshape(self.batch_r[0:self.batch_size], [1, -1]), [1, 0])
		self.negative_h = tf.transpose(tf.reshape(self.batch_h[self.batch_size:self.batch_seq_size], [self.n_negative, -1]), perm=[1, 0])
		self.negative_t = tf.transpose(tf.reshape(self.batch_t[self.batch_size:self.batch_seq_size], [self.n_negative, -1]), perm=[1, 0])
		self.negative_r = tf.transpose(tf.reshape(self.batch_r[self.batch_size:self.batch_seq_size], [self.n_negative, -1]), perm=[1, 0])
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
