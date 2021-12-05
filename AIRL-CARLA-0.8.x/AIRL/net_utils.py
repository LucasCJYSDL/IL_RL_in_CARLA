import tensorflow as tf
from contextlib import contextmanager
from baselines.common import colorize
import baselines.common.tf_util as U
import time
import numpy as np


def weight_xavi_init(shape, name):
    return tf.get_variable(shape=shape, name=name, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape, name):
    return tf.get_variable(shape=shape, name=name, initializer=tf.constant_initializer(0.1))


def fc(x, output_size, layer_name, activation="relu"):
    filters_in = x.get_shape()[-1]
    shape = [filters_in, output_size]
    weights = weight_xavi_init(shape, layer_name + "_w")
    bias = bias_variable([output_size], layer_name + "_b")
    if activation == "identity":
        return tf.identity(tf.nn.xw_plus_b(x, weights, bias))
    return tf.nn.relu(tf.nn.xw_plus_b(x, weights, bias))

@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))

def intprod(x):
    return U.intprod(x)

def flatgrad(loss, var_list, clip_norm=None):
    return U.flatgrad(loss, var_list, clip_norm)

class GetFlat(object):
    def __init__(self, var_list, sess):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [U.numel(v)]) for v in var_list])
        self.sess = sess

    def __call__(self):
        return self.sess.run(self.op)

class SetFromFlat(object):
    def __init__(self, var_list, sess, dtype=tf.float32):
        assigns = []
        shapes = list(map(U.var_shape, var_list))
        total_size = np.sum([U.intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = U.intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)
        self.sess = sess

    def __call__(self, theta):
        self.sess.run(self.op, feed_dict={self.theta: theta})

def function(inputs, outputs, sess, updates=None, givens=None):

    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens, sess=sess)
    elif isinstance(outputs, (dict, U.collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens, sess=sess)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens, sess=sess)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens, sess):
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (type(inpt) is tf.Tensor and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.inputs = inputs
        self.input_names = {inp.name.split("/")[-1].split(":")[0]: inp for inp in inputs}
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens
        self.sess = sess

    def _feed_input(self, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = U.adjust_shape(inpt, value)

    def __call__(self, *args, **kwargs):
        assert len(args) + len(kwargs) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = U.adjust_shape(inpt, feed_dict.get(inpt, self.givens[inpt]))
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        for inpt_name, value in kwargs.items():
            self._feed_input(feed_dict, self.input_names[inpt_name], value)
        results = self.sess.run(self.outputs_update, feed_dict=feed_dict)[:-1]
        return results