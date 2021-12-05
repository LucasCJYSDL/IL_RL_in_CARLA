import tensorflow  as  tf

def droupout(x, droup, layer_name):
    return tf.nn.dropout(x, droup,  name=layer_name + '_dropout')


def weight_xavi_init(shape,name):
    return tf.get_variable(shape=shape,name=name,initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape,name):
    return tf.get_variable(shape=shape,name=name,initializer=tf.constant_initializer(0.1))

def fc(x, output_size, layer_name, activation_fn):
    assert len(x.shape) == 2

    filters_in = x.get_shape()[-1]
    shape = [filters_in, output_size]
    weights = weight_xavi_init(shape,layer_name+"_w")
    bias = bias_variable([output_size],layer_name+"_b")

    if activation_fn == "relu":
        return tf.nn.relu(tf.nn.xw_plus_b(x, weights,bias))
    elif activation_fn == "tanh":
        return tf.nn.tanh(tf.nn.xw_plus_b(x, weights,bias))
    elif activation_fn == "identity":
        return tf.identity(tf.nn.xw_plus_b(x, weights,bias))