import tensorflow as tf
import numpy as np
from .net_utils import fc

class ValueNet(object):
    def __init__(self, sess, learning_rate):
        self.sess = sess
        self.learning_rate = learning_rate
        self.image_input, self.speed_input, self.branches = self.create_value_network(scope="Value")

        self.loss_name = [["loss_follow"], ["loss_left"], ["loss_right"], ["loss_straight"]]
        self.losses = []
        self.ret = tf.placeholder(tf.float32, [None, 1])
        self.branch_input = tf.placeholder(tf.float32, [None, 1])
        self.branch_optimizes = [self.get_branch_optimize(i) for i in range(4)]

    def create_value_network(self, scope):
        branches = []  # 4 branches:follow,straight,turnLeft,turnRight
        with tf.variable_scope(scope):
            with tf.variable_scope("Share"):
                image_input = tf.placeholder(dtype=tf.float32, shape=(None, 512), name="ImageInput")
                speed_input = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="SpeedInput")
                speed_fc1 = fc(speed_input, 128, "speed_layer_1")
                speed_fc2 = fc(speed_fc1, 128, "speed_layer_2")
                x_fc = tf.concat([image_input, speed_fc2], 1)
                x_fc = fc(x_fc, 512, "concat_fc")
            for i in range(4):
                scope_name = "branch_{}".format(i)
                with tf.name_scope(scope_name):
                    branch_output = fc(x_fc, 256, scope_name + "_layer1")
                    branch_output = fc(branch_output, 256, scope_name + "_layer2")
                    branch_output = fc(branch_output, 1, scope_name + "_out")
                branches.append(branch_output)
        return image_input, speed_input, branches

    def get_weights(self, scope):
        all_weights = []
        params_fc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        all_weights.extend(params_fc)
        return all_weights

    # Share weights + one branch weights
    def get_weights_branch(self, branch_num):
        branch_weights = []
        params_share = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Value/Share")
        scope_name = scope_name = "Value/branch_{}".format(branch_num)
        params_branch = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
        branch_weights.extend(params_share)
        branch_weights.extend(params_branch)
        return branch_weights

    def get_branch_optimize(self, branch_num):
        branch_out = self.branches[branch_num]
        branch_params = self.get_weights_branch(branch_num)
        loss = tf.reduce_mean(tf.squared_difference(self.ret, branch_out))
        self.losses.append(loss)
        branch_optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=branch_params)
        return branch_optimize

    def train_branch(self, image_input, speed_input, ret, branch_num):
        ret = self.sess.run([self.branch_optimizes[branch_num], self.losses[branch_num]], feed_dict={
            self.image_input: np.reshape(image_input, (-1, 512)),
            self.speed_input: np.reshape(speed_input, (-1, 1)),
            self.ret: np.reshape(ret, (-1, 1))
        })
        return ret[-1]

    def pridect_value(self, image_input, speed_input, branch_num):
        value = self.sess.run(self.branches[branch_num], feed_dict={
            self.image_input: np.reshape(image_input, (-1, 512)),
            self.speed_input: np.reshape(speed_input, (-1, 1))
        })
        return value
