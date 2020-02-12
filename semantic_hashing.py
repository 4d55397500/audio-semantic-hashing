import numpy as np
import tensorflow as tf
from collections import Counter

from constants import INTERMEDIATE_LAYER_DIMS, INITIAL_NOISE_STD,\
    NOISE_MULTIPLICATIVE_INCREMENT


class SemanticHashing(object):

    """ Semantic hashing with dense layers """

    def __init__(self, xdim, hdim):

        layer_dims = [xdim] + INTERMEDIATE_LAYER_DIMS + [hdim]

        self.xdim = xdim
        self.hdim = hdim

        self.noise = tf.compat.v1.placeholder('float', shape=[None, hdim])
        self.x_in = tf.compat.v1.placeholder('float', [None, xdim])

        self.encoder_layers = [self.x_in]
        for i in range(len(layer_dims) - 1):
            if i == len(layer_dims) - 2:
                layer = tf.nn.sigmoid(tf.add(
                    tf.matmul(self.encoder_layers[i],
                              tf.Variable(tf.compat.v1.random_normal([layer_dims[i], layer_dims[i + 1]]))),
                    tf.Variable(tf.compat.v1.random_normal([layer_dims[i + 1]]))) + self.noise)
                self.encoder_layers.append(layer)
            else:
                layer = tf.add(
                    tf.matmul(self.encoder_layers[i],
                              tf.Variable(tf.compat.v1.random_normal([layer_dims[i], layer_dims[i + 1]]))),
                    tf.Variable(tf.compat.v1.random_normal([layer_dims[i + 1]])))
                self.encoder_layers.append(layer)

        self.h = self.encoder_layers[-1]

        self.decoder_layers = [self.h]
        for i in range(len(layer_dims) - 1):
            layer = tf.add(
                tf.matmul(self.decoder_layers[i],
                          tf.Variable(tf.compat.v1.random_normal([layer_dims[::-1][i], layer_dims[::-1][i + 1]]))),
                tf.Variable(tf.compat.v1.random_normal([layer_dims[::-1][i + 1]])))
            self.decoder_layers.append(layer)

        self.output = self.decoder_layers[-1]
        self.x_out = tf.compat.v1.placeholder('float', [None, xdim])
        self.loss = tf.reduce_mean(tf.square(self.x_out - self.output))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(0.1).minimize(self.loss)
        self.initializer = tf.compat.v1.global_variables_initializer()
        self.session = tf.compat.v1.Session()

    def train(self, x_train, batch_size, n_epochs):

        self.session.run(self.initializer)
        noise_std = INITIAL_NOISE_STD
        for epoch in range(n_epochs):
            epoch_loss = 0.
            noise_std *= NOISE_MULTIPLICATIVE_INCREMENT
            noise = np.random.normal(scale=noise_std, size=(batch_size, self.hdim))
            for i in range(int(x_train.shape[0] / batch_size)):
                batch = x_train[i * batch_size: (i + 1) * batch_size]
                _, batch_loss = self.session.run([self.optimizer, self.loss],
                                                 feed_dict={self.x_in: batch, self.x_out: batch, self.noise: noise})
                epoch_loss += batch_loss
            print(f"Epoch: {epoch}/{n_epochs} Epoch loss: {epoch_loss} Noise std: {noise_std}")
            h_entropy = self.encoded_entropy(x_train)
            print(f"Encoded entropy: {h_entropy}")

    def encode(self, x):
        return self.session.run(
            self.h,
            feed_dict={self.x_in: x,
                       self.noise: np.random.normal(0.0, 0.0,
                                                    size=(x.shape[0], self.hdim))
                       })

    def encoded_entropy(self, x):
        h = self.encode(x)
        bits = [''.join([str(e) for e in b]) for b in h]
        cts = Counter(bits).values()
        ps = [ct * 1.0 / sum(cts) for ct in cts]
        return sum([-p * np.log(p) for p in ps])

    def decode(self, x):
        return self.session.run(self.output, feed_dict={self.x_in: x})


