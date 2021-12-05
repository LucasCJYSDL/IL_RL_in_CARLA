"""
The critic layer, maps (states, action) into Q-function value
"""


import numpy as np
import tensorflow as  tf


class Critic:

    def __init__(self, sess, preprocessor, branch_list):
        self.sess = sess
        self.branch_list = branch_list
        
        self.image_ph, self.lidar_ph, self.measure_ph, self.dropout = preprocessor.get_input()
        self.feat = preprocessor.get_feat()

        self.outputs, self.action_ph = self.create_graph("Critic")
        self.target_outputs, self.target_action_ph = self.create_graph("TargetCritic")

    def set_training(self, args):
        self.get_assign_target()
        self.get_action_gradients()
        self.get_rl_optimizers(args.rl_lr_c)
    

    # NN Architecture
    def create_graph(self, scope):
        initializer = tf.truncated_normal_initializer(stddev=0.01)

        outputs = {}
        action_ph = tf.placeholder(dtype=tf.float32, shape=(None, 2))

        with tf.variable_scope(scope):
            with tf.variable_scope("Share"):
                fc0 = tf.layers.dense(action_ph, units=64, activation=tf.nn.relu, kernel_initializer=initializer)
                cat1 = tf.concat([fc0, self.feat], axis=1)

                fc1 = tf.layers.dense(cat1, units=512, activation=tf.nn.relu, kernel_initializer=initializer)
                drop1 = tf.nn.dropout(fc1, self.dropout[0])

            for i in self.branch_list:
                with tf.variable_scope("Branch_" + str(i)):
                    fci1 = tf.layers.dense(drop1, units=256, activation=tf.nn.relu, kernel_initializer=initializer)
                    dpi1 = tf.nn.dropout(fci1, self.dropout[0])

                    fci2 = tf.layers.dense(dpi1, units=128, activation=tf.nn.relu, kernel_initializer=initializer)
                    dpi2 = tf.nn.dropout(fci2, self.dropout[0])
                    #dpi2 = dpi1
                    outputs[i] = tf.layers.dense(dpi2, units=1, kernel_initializer=initializer) # shouldn't take activation

        return outputs, action_ph
    

    # Assign Actor to Target Actor
    def get_assign_target(self):
        self.tau_ph  = tf.placeholder(dtype= tf.float32, shape= 1)

        param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic")
        param_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="TargetCritic")

        self.assign_target = [
            tf.assign(t, (1 - self.tau_ph[0]) * t + self.tau_ph[0] * p) for t, p in zip(param_target, param)
        ]
    
    def run_assign_target(self, tau):
        """If tau = 1, then copy critic to target critic"""
        self.sess.run(
            self.assign_target,
            feed_dict={
                self.tau_ph : [tau]
            }
        )
    

    # Action Gradient
    def get_action_gradients(self):
        self.action_gradients = {}
        for i in self.branch_list:
            self.action_gradients[i] = tf.gradients(self.outputs[i], self.action_ph)

    def run_action_gradients(self, images, lidars, measures, actions, branch, dropout):
        return self.sess.run(
            self.action_gradients[branch],
            feed_dict={
                self.image_ph: images,
                self.lidar_ph: lidars,
                self.measure_ph: measures,
                self.action_ph: actions,
                self.dropout: [dropout],
            }
        )[0]


    # RL optimizer
    def get_rl_optimizers(self, lr):
        self.label_ph = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        self.rl_optimizers = {}
        for i in self.branch_list:
            params = []
            #params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Preprocessor") 
            params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/Share")
            params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/Branch_" + str(i))

            output = self.outputs[i]

            loss = tf.reduce_mean(tf.square(self.label_ph - output))

            self.rl_optimizers[i] = [
                loss, output, tf.train.AdamOptimizer(lr).minimize(loss, var_list=params)
            ]

    def run_rl_optimizers(self, images, lidars, measures, actions, labels, branch, dropout):
        loss, out, _ = self.sess.run(
            self.rl_optimizers[branch],
            feed_dict = {
                self.image_ph: images,
                self.lidar_ph: lidars,
                self.measure_ph: measures,
                self.action_ph: actions,
                self.label_ph: labels,
                self.dropout: [dropout],
            }
        )
        return loss, out
    

    # predict
    def predict_target_q(self, images, lidars, measures, actions, branch, dropout):
        return self.sess.run(
            self.target_outputs[branch],
            feed_dict={
                self.image_ph: images,
                self.lidar_ph: lidars,
                self.measure_ph: measures,
                self.target_action_ph: actions,
                self.dropout: [dropout],
            }
        )


