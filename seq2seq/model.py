# coding: utf-8

import os
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from seq2seq.utils import feed

tf.reset_default_graph()
sess = tf.InteractiveSession()

class Seq2seq:
    """
    """

    def __init__(self, 
                 vocab_size=10,
                 embedding_size=20, 
                 encoder_hidden_units=20, 
                 decoder_hidden_units=20,
                 weight_file="model_weight.h5"):

        # Define model hyperparameters.
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        self.weight_file = weight_file

        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
        self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

        # Embedding.
        embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], minval=-1.0, maxval=1.0), 
                                dtype=tf.float32)
        encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.decoder_inputs)

        # Define encoder cell
        encoder_cell = LSTMCell(encoder_hidden_units)
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                 inputs=encoder_inputs_embedded, 
                                                                 dtype=tf.float32, 
                                                                 time_major=True)

        # Define decoder cell
        decoder_cell = LSTMCell(decoder_hidden_units)
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(cell=decoder_cell,
                                                                 inputs=decoder_inputs_embedded, 
                                                                 initial_state=encoder_final_state,
                                                                 dtype=tf.float32, 
                                                                 time_major=True,
                                                                 scope='plain_decoder')

        # Define projection layer
        decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
        self.decoder_prediction = tf.argmax(decoder_logits, axis=2)  # [max_time, batch_size, hidden_units]

        # Define optimzer.
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=vocab_size, dtype=tf.float32), 
            logits=decoder_logits)

        # Calculate loss.
        self.loss = tf.reduce_mean(stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, batches, num_epochs=12, batch_size=64, lr=1e-3, max_num_batches=5001):
        """ Note: num_epochs isn't using here. 
            Size of batches feeded is determined by max_num_batches.
        """
        loss_history = []

        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for step in range(max_num_batches):
            fd = feed(batches)
            _feed_dict = {self.encoder_inputs: fd['encoder_inputs'], 
                          self.decoder_inputs: fd['decoder_inputs'],
                          self.decoder_targets: fd['decoder_targets']}
            _, loss = sess.run([self.train_op, self.loss], feed_dict=_feed_dict)
            loss_history.append(loss)

            if step == 0 or step % 500 == 0:
                print('batch:', step)
                print('mini batch loss:', sess.run(self.loss, 
                                                   feed_dict=_feed_dict))
                predict_ = sess.run(self.decoder_prediction, feed_dict=_feed_dict)

                for i, (input_, pred_) in enumerate(zip(_feed_dict[self.encoder_inputs].T, predict_.T)):
                    print('sample:', i+1)
                    print('input:\n', input_)
                    print('predict:\n', pred_)
                    if i >= 2:
                        break
                print()

        save_dir = os.path.join('.', 'checkpoints')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, self.weight_file)
        saver.save(sess=sess, save_path=save_path)
        sess.close()
        return loss_history


    def predict(self, sequence):
        saver = tf.train.Saver()
        sess = tf.Session()
        load_path = os.path.join('./checkpoints', self.weight_file)
        saver.restore(sess, load_path)

        fd = feed([sequence])
        _feed_dict = {self.encoder_inputs: fd['encoder_inputs'], 
                      self.decoder_inputs: fd['decoder_inputs'],
                      self.decoder_targets: fd['decoder_targets']}
        predict_ = sess.run(self.decoder_prediction, feed_dict=_feed_dict)
        sess.close()
        return predict_.T

    def __str__(self):
        attrs = []
        for key in self.__dict__:
            attrs.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        return ', '.join(attrs)

    def __repr__(self):
        return self.__str__()






