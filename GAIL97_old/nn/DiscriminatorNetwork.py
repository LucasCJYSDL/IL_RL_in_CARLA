import tensorflow  as  tf
import numpy as  np
from collections import OrderedDict
from nn.layers import fc

def logsigmoid(a):
    return -tf.nn.softplus(-a)

def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class Discriminator(object):
    def __init__(self, sess, args, branch_list):
        self.sess = sess
        self.learning_rate = args.lr_d
        self.entcoeff = args.entcoeff
        self.branch_list = branch_list
        self.hid_dim = args.hid_dim
        self.build_ph(scope="Discriminator")
        self.branches = self.create_discriminator_network(self.image_input, self.lidar_input, self.measure_input, self.action_input, scope="Discriminator")
        self.exp_branches = self.create_discriminator_network(self.exp_image_input, self.exp_lidar_input, self.exp_measure_input, self.exp_action_input, scope="Discriminator", reuse=True)

        self.rewards = OrderedDict()
        self.branch_optimizes = OrderedDict()
        for i in self.branch_list:
            self.branch_optimizes[i] = self.get_branch_optimize(i)

    def build_ph(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("Share"):
                self.image_input = tf.placeholder(tf.float32, (None, 512), name="ImageInput")
                self.lidar_input = tf.placeholder(dtype=tf.float32, shape=(None, 360), name="LidarInput")
                self.measure_input = tf.placeholder(dtype=tf.float32, shape=(None, 7), name="MeasureInput")
                self.action_input = tf.placeholder(tf.float32, (None, 2), name="ActionInput")

                self.exp_image_input = tf.placeholder(tf.float32, (None, 512), name="ExpImageInput")
                self.exp_lidar_input = tf.placeholder(dtype=tf.float32, shape=(None, 360), name="ExpLidarInput")
                self.exp_measure_input = tf.placeholder(dtype=tf.float32, shape=(None, 7), name="ExpMeasureInput")
                self.exp_action_input = tf.placeholder(tf.float32, (None, 2), name="ExpActionInput")

    def create_discriminator_network(self, image_input, lidar_input, measure_input, action_input, scope, reuse=False):
        branches = OrderedDict()
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("Share"):

                lidar_fc1 = fc(lidar_input, self.hid_dim, "lidar_layer_1", activation_fn='tanh')
                measure_fc1 = fc(measure_input, self.hid_dim, "measure_layer_1", activation_fn='tanh')
                action_fc1 = fc(action_input, self.hid_dim, "action_layer_1", activation_fn="tanh")

                x_fc = tf.concat([image_input, lidar_fc1, measure_fc1, action_fc1], 1)
                x_fc = fc(x_fc, 512, "concat_fc", activation_fn="tanh")

            for i in self.branch_list:
                scope_name = "branch_{}".format(i)
                with tf.name_scope(scope_name):
                    branch_output = fc(x_fc, self.hid_dim, scope_name + "_layer1", activation_fn="tanh")
                    branch_output = fc(branch_output, self.hid_dim, scope_name + "_layer2", activation_fn='tanh')
                    branch_output = fc(branch_output, 1, scope_name + "_out", activation_fn="identity")
                branches[i] = branch_output
        return branches


    def get_weights(self, scope):
        all_weights = []
        params_fc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        all_weights.extend(params_fc)
        return all_weights

    def get_weights_branch(self, branch_num):
        branch_weights = []
        params_share = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator/Share")
        scope_name = "Discriminator/branch_{}".format(branch_num)
        params_branch = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
        branch_weights.extend(params_share)
        branch_weights.extend(params_branch)
        return branch_weights

    def get_branch_optimize(self, branch_num):
        branch_out = self.branches[branch_num]
        exp_branch_out = self.exp_branches[branch_num]
        branch_params = self.get_weights_branch(branch_num)
        reward_op = -tf.log(1-tf.nn.sigmoid(branch_out)+1e-8)
        self.rewards[branch_num] = reward_op

        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=branch_out, labels=tf.zeros_like(branch_out))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=exp_branch_out, labels=tf.ones_like(exp_branch_out))
        expert_loss = tf.reduce_mean(expert_loss)

        logits = tf.concat([branch_out, exp_branch_out], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -self.entcoeff * entropy

        total_loss = generator_loss+expert_loss+entropy_loss


        return [total_loss, tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss, var_list=branch_params)]


    def train_branch(self, image_input, lidar_input, measure_input, action_input, exp_image_input, exp_lidar_input, exp_measure_input, exp_action_input, branch_num):
        loss, _ = self.sess.run(self.branch_optimizes[branch_num], feed_dict={
            self.image_input: np.reshape(image_input, (-1, 512)),
            self.lidar_input: np.reshape(lidar_input, (-1, 360)),
            self.measure_input: np.reshape(measure_input, (-1, 7)),
            self.action_input: np.reshape(action_input, (-1, 2)),
            self.exp_image_input: np.reshape(exp_image_input, (-1, 512)),
            self.exp_lidar_input: np.reshape(exp_lidar_input, (-1, 360)),
            self.exp_measure_input: np.reshape(exp_measure_input, (-1, 7)),
            self.exp_action_input: np.reshape(exp_action_input, (-1, 2)),
        }) #???

        return loss

    def predict_reward(self, image_input, lidar_input, measure_input, action_input, branch_num):

        return self.sess.run(self.rewards[branch_num], feed_dict={
            self.image_input: np.reshape(image_input, (-1, 512)),
            self.lidar_input: np.reshape(lidar_input, (-1, 360)),
            self.measure_input: np.reshape(measure_input, (-1, 7)),
            self.action_input: np.reshape(action_input, (-1, 2)),
        })