import tensorflow as tf
import OpenKE

data = OpenKE.DataLoader(data_path='./benchmarks/WN18RR')

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        model = OpenKE.models.Model.load('output/model.pickle')

        # Test the model
        test_placeholders = {
            'h': tf.placeholder(tf.int64, [None]),
            't': tf.placeholder(tf.int64, [None]),
            'r': tf.placeholder(tf.int64, [None]),
        }
        predict = model.predict(
                test_placeholders['h'],
                test_placeholders['t'],
                test_placeholders['r'])

        def predict_fn(h,t,r):
            return sess.run(predict, {
                    test_placeholders['h']:h,
                    test_placeholders['t']:t,
                    test_placeholders['r']:r})

        data.test_link_prediction(predict_fn)
        data.test_triple_classification(predict_fn)
