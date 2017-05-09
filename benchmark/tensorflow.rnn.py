#!/usr/bin/env python
import time
import optparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib
from tensorflow.python.ops import rnn


def get_feed_dict(x_data, y_data=None):
    feed_dict = {}

    if y_data is not None:
        feed_dict[y] = y_data

    for i in xrange(x_data.shape[0]):
        feed_dict[x[i]] = x_data[i, :, :]

    return feed_dict


# Parameters
optparser = optparse.OptionParser()
optparser.add_option("--mode", default='inference', type=str, help="inference or train")
optparser.add_option("--network_type", default='rnn', type=str, help="Network type (rnn, lstm)")
optparser.add_option("--hidden_size", default=128, type='int', help="Hidden layer size")
optparser.add_option("--seq_length", default=16, type='int', help="Sequence length")
optparser.add_option("--batch_size", default=32, type='int', help="Batch size")
optparser.add_option("--num_batches", default=64, type='int', help="Number of batches")
optparser.add_option("--num_iter", default=5, type='int', help="Number of iterations")
opts = optparser.parse_args()[0]

network_type = opts.network_type
hidden_size = opts.hidden_size
hidden_size = opts.hidden_size
seq_length = opts.seq_length
batch_size = opts.batch_size
n_batch = opts.num_batches
n_iter = opts.num_iter
n_samples = batch_size * n_batch

# Data
xinput = np.random.rand(seq_length, batch_size, hidden_size, n_batch).astype(np.float32)
ytarget = np.random.rand(batch_size, hidden_size, n_batch).astype(np.float32)

x = tf.placeholder(tf.float32, [seq_length, batch_size, hidden_size], name="x") #for i in range(seq_length)]
y = tf.placeholder(tf.float32, [batch_size, hidden_size], name="y")

if network_type == 'rnn':
    cell = tensorflow.contrib.rnn.BasicRNNCell(hidden_size)
elif network_type == 'lstm':
    cell = tensorflow.contrib.rnn.BasicLSTMCell(hidden_size)
else:
    raise Exception('Unknown network! '+network_type)

output, _cell_state = rnn.dynamic_rnn(cell, x, dtype=tf.float32)
cost = tf.reduce_sum((output[-1] - y) ** 2)

optim = tf.train.GradientDescentOptimizer(0.01)
train_op = optim.minimize(cost)

func = output if opts.mode.lower() == 'inference' else train_op

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    # Warmup
    session.run(func, feed_dict={
        x: xinput[:,:,:,0],
        y: ytarget[:,:,0]
    })
    elapsed = 0
    for j in xrange(n_iter):
        print('Iter %d' % j)
        start = time.time()
        for i in xrange(0, n_batch):
            session.run(func, feed_dict={
                x: xinput[:,:,:,i],
                y: ytarget[:,:,i]
            })
        elapsed = elapsed + (time.time() - start)
    elapsed = elapsed / n_iter
    samples_per_sec = n_samples / elapsed
    print(opts)
    print("--- %i samples in %4.4f seconds (%4.2f samples/s, %4.4f ms/sample) ---" %
          (n_samples, elapsed, samples_per_sec, 1000 / samples_per_sec))
