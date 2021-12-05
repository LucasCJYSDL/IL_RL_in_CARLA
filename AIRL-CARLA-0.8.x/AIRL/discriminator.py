import tensorflow  as  tf
import numpy as  np
from .net_utils import fc

class Discriminator(object):
    def __init__(self, sess, learning_rate):
        self.sess = sess
        self.learning_rate = learning_rate
        self.build_ph(scope="Discriminator")
        self.branches = self.create_discriminator_network(self.image_input, self.speed_input, self.action_input,
                                                          scope="Discriminator")
        self.exp_branches = self.create_discriminator_network(self.exp_image_input, self.exp_speed_input,
                                                              self.exp_action_input, scope="Discriminator", reuse=True)

        self.loss_name = [["loss_follow"], ["loss_left"], ["loss_right"], ["loss_straight"]]
        self.losses = []
        self.d_taus = []
        self.branch_optimizes = [self.get_branch_optimize(i) for i in range(4)]

        self.rewards = [self.get_reward(dt) for dt in self.d_taus]

    def build_ph(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("Share"):
                self.image_input = tf.placeholder(tf.float32, (None, 512), name="ImageInput")
                self.speed_input = tf.placeholder(tf.float32, (None, 1), name="SpeedInput")
                self.action_input = tf.placeholder(tf.float32, shape=(None, 3), name="ActionInput")

                self.exp_image_input = tf.placeholder(tf.float32, (None, 512), name="ExpImageInput")
                self.exp_speed_input = tf.placeholder(tf.float32, (None, 1), name="ExpSpeedInput")
                self.exp_action_input = tf.placeholder(tf.float32, shape=(None, 3), name="ExpActionInput")

                self.labels_ph = tf.placeholder(tf.float32, (None, 1), name='labels_ph')
                self.lprobs_ph = tf.placeholder(tf.float32, (None, 1), name='log_probs_ph')

    def create_discriminator_network(self, image_input, speed_input, action_input, scope, reuse=False):
        branches = []  # 4 branches:follow,straight,turnLeft,turnRight
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("Share"):

                speed_fc1 = fc(speed_input, 128, "speed_layer_1")
                speed_fc2 = fc(speed_fc1, 128, "speed_layer_2")

                action_fc1 = fc(action_input, 128, "action_layer_1")
                action_fc2 = fc(action_fc1, 128, "action_layer_2")

                x_fc = tf.concat([image_input, speed_fc2, action_fc2], 1)
                x_fc = fc(x_fc, 512, "concat_fc")
            for i in range(4):
                scope_name = "branch_{}".format(i)
                with tf.name_scope(scope_name):
                    branch_output = fc(x_fc, 256, scope_name + "_layer1")
                    branch_output = fc(branch_output, 256, scope_name + "_layer2")
                    branch_output = fc(branch_output, 1, scope_name + "_out", activation="identity")
                branches.append(branch_output)
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
        energy = tf.concat([branch_out, exp_branch_out], axis=0)
        log_p = -energy
        log_q = self.lprobs_ph
        log_pq = tf.reduce_logsumexp([log_p, log_q], axis=0)
        d_tau = tf.exp(log_p - log_pq)
        self.d_taus.append(d_tau)
        loss = -tf.reduce_mean(self.labels_ph * (log_p - log_pq) + (1 - self.labels_ph) * (log_q - log_pq))
        self.losses.append(loss)
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=branch_params)

    def get_reward(self, d_tau):
        reward, _ = tf.split(axis=0, num_or_size_splits=2, value=tf.log(d_tau + 1e-8) - tf.log(1 - d_tau + 1e-8))
        return reward

    def train_branch(self, image_input, speed_input, action_input, exp_image_input, exp_speed_input, exp_action_input,
                     labels, lprobs, branch_num):
        ret = self.sess.run([self.branch_optimizes[branch_num], self.losses[branch_num]], feed_dict={
            self.image_input: np.reshape(image_input, (-1, 512)),
            self.speed_input: np.reshape(speed_input, (-1, 1)),
            self.action_input: np.reshape(action_input, (-1, 3)),
            self.exp_image_input: np.reshape(exp_image_input, (-1, 512)),
            self.exp_speed_input: np.reshape(exp_speed_input, (-1, 1)),
            self.exp_action_input: np.reshape(exp_action_input, (-1, 3)),
            self.labels_ph: np.reshape(labels, (-1, 1)),
            self.lprobs_ph: np.reshape(lprobs, (-1, 1))
        }) #???

        return ret[-1]

    def predict_reward(self, image_input, speed_input, action_input, lprobs, branch_num):

        exp_image_input = np.zeros_like(image_input)
        exp_speed_input = np.zeros_like(speed_input)
        exp_action_input = np.zeros_like(action_input)
        batch_size = len(action_input)
        labels = np.zeros((batch_size * 2, 1))
        labels[batch_size:] = 1.0

        lprobs_exp = np.ones_like(lprobs)
        lprobs = np.concatenate([lprobs, lprobs_exp], axis=0)

        return self.sess.run(self.rewards[branch_num], feed_dict={
            self.image_input: np.reshape(image_input, (-1, 512)),
            self.speed_input: np.reshape(speed_input, (-1, 1)),
            self.action_input: np.reshape(action_input, (-1, 3)),
            self.exp_image_input: np.reshape(exp_image_input, (-1, 512)),
            self.exp_speed_input: np.reshape(exp_speed_input, (-1, 1)),
            self.exp_action_input: np.reshape(exp_action_input, (-1, 3)),
            self.labels_ph: np.reshape(labels, (-1, 1)),
            self.lprobs_ph: np.reshape(lprobs, (-1, 1))
        })
