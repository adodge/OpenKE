import OpenKE
import tensorflow as tf
import os

n_epochs = 1000
optimizer_alpha = 0.001
model_class = OpenKE.models.TransE

save_dir = 'output'
save_steps = 100

os.makedirs(save_dir, exist_ok=True)

data = OpenKE.DataLoader(data_path='./benchmarks/WN18RR')

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():

        # Create the model and inputs
        input_placeholders = {
            'h': tf.placeholder(tf.int64, [data.batch_seq_size]),
            't': tf.placeholder(tf.int64, [data.batch_seq_size]),
            'r': tf.placeholder(tf.int64, [data.batch_seq_size]),
            'y': tf.placeholder(tf.float32, [data.batch_seq_size]),
        }

        initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        with tf.variable_scope('model'):
            model = model_class(
                    n_entities=data.n_entities,
                    n_relations=data.n_relations)

            print(model.arguments)

            loss = model.loss(
                    input_placeholders['h'],
                    input_placeholders['t'],
                    input_placeholders['r'],
                    input_placeholders['y'],
                    data.batch_size, data.n_negative)
            
            optimizer = tf.train.GradientDescentOptimizer(optimizer_alpha)
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars)
        saver = tf.train.Saver()
        sess.run(tf.initializers.global_variables())

        # Fit the model
        for epoch in range(n_epochs):
            res = 0.0
            for batch in range(data.n_batches):
                h,t,r,y = data.sample()
                feed_dict = {
                        input_placeholders['h']: h,
                        input_placeholders['t']: t,
                        input_placeholders['r']: r,
                        input_placeholders['y']: y,
                }
                _,epoch_loss = sess.run([train_op, loss], feed_dict)
                res += epoch_loss

            print(epoch, res)
            if save_steps and epoch % save_steps == 0:
                saver.save(sess, os.path.join(save_dir, 'intermediate.epoch%d.tf' % epoch))
        
        saver.save(sess, os.path.join(save_dir, 'final.tf'))

        # Save model parameters
        model.dump(os.path.join(save_dir, "model.pickle"))
