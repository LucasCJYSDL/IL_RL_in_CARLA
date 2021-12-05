import tensorflow as tf
import numpy as np
from .net_utils import fc
from baselines.common.distributions import DiagGaussianPdType
from .trpo import TrpoSolver


class Policy(object):
    def __init__(self, sess, min_std=1e-6, entcoeff=0.0):
        self.sess = sess
        self.entcoeff = entcoeff
        self.pdtype = DiagGaussianPdType(3)
        self.min_std = min_std
        self.image_input = tf.placeholder(dtype=tf.float32, shape=(None, 512), name="PolicyImageInput")
        self.speed_input = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="PolicySpeedInput")
        self.means, self.logstds, self.pds, self.acs = self.create_policy_network(self.image_input, self.speed_input, scope="Policy")
        _, _, self.oldpds, _ = self.create_policy_network(self.image_input, self.speed_input, scope="OldPolicy")
        self.assign_old = self.get_assign_old()

        self.loss_name = [["loss_follow"], ["loss_left"], ["loss_right"], ["loss_straight"]]
        self.branch_input = tf.placeholder(tf.float32, [None, 1])
        self.branch_optimizes = [self.get_branch_optimize(i) for i in range(4)]

    def create_policy_network(self, image_input, speed_input, scope):
        means, logstds, pds, acs = [], [], [], []  # 4 branches:follow,straight,turnLeft,turnRight
        with tf.variable_scope(scope):
            with tf.variable_scope("Share"):
                speed_fc1 = fc(speed_input, 128, "speed_layer_1")
                speed_fc2 = fc(speed_fc1, 128, "speed_layer_2")
                x_fc = tf.concat([image_input, speed_fc2], 1)
                x_fc = fc(x_fc, 512, "concat_fc")
            for i in range(4):
                scope_name = "branch_{}".format(i)
                with tf.name_scope(scope_name):
                    branch_output = fc(x_fc, 256, scope_name + "_layer1")
                    branch_output = fc(branch_output, 256, scope_name + "_layer2")
                    mean = fc(branch_output, 3, scope_name + "_out", activation="identity")
                    logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0]//2],
                                             initializer=tf.constant_initializer(np.log(0.1)))
                    reg_logstd = tf.maximum(logstd, np.log(self.min_std))
                    flat = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                    pd = self.pdtype.pdfromflat(flat)
                    ac = pd.sample()
                means.append(mean)
                logstds.append(reg_logstd)
                pds.append(pd)
                acs.append(ac)
        return means, logstds, pds, acs

    def get_weights(self, scope):
        all_weights = []
        params_fc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        all_weights.extend(params_fc)
        return all_weights

    # Share weights + one branch weights
    def get_weights_branch(self, branch_num):
        branch_weights = []
        params_share = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Policy/Share")
        scope_name = "Policy/branch_{}".format(branch_num)
        params_branch = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
        branch_weights.extend(params_share)
        branch_weights.extend(params_branch)
        return branch_weights

    def get_branch_optimize(self, branch_num):
        branch_params = self.get_weights_branch(branch_num)
        scope_name = "Policy/branch_{}".format(branch_num)
        with tf.name_scope(scope_name):
            ac_input = self.pdtype.sample_placeholder([-1], "ac")
            branch_optimize = TrpoSolver(self.sess, self.pds[branch_num], self.oldpds[branch_num], self.image_input,
                                         self.speed_input, ac_input, branch_params, self.entcoeff)
        return branch_optimize

    def get_assign_old(self):
        params = self.get_weights(scope="Policy")
        old_params = self.get_weights(scope="OldPolicy")
        return [tf.assign(old, cur) for old, cur in zip(old_params, params)]

    def assign_old_cur(self):
        self.sess.run(self.assign_old)

    def train_branch(self, image_input, speed_input, ac, atarg, cg_damping, max_kl, cg_iters, branch_num):
        args = image_input, speed_input, ac, atarg
        return self.branch_optimizes[branch_num].run(args, self.assign_old_cur, cg_damping, max_kl, cg_iters)

    def pridect_output(self, image_input, speed_input, branch_num):
        output = self.sess.run([self.acs[branch_num], self.means[branch_num], self.logstds[branch_num]], feed_dict={
            self.image_input: np.reshape(image_input, (-1, 512)),
            self.speed_input: np.reshape(speed_input, (-1, 1))
        })
        return output[0], output[1], output[2]

