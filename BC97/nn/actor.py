"""
The actor layer, maps states into actions
"""


import numpy as np
import tensorflow as  tf


class Actor:

    def __init__(self, sess, preprocessor, branch_list):
        self.sess = sess
        self.branch_list = branch_list
        
        self.image_ph, self.lidar_ph, self.measure_ph, self.dropout = preprocessor.get_input()
        self.feat = preprocessor.get_feat()

        self.outputs = self.create_graph("Actor")        
        self.target_outputs = self.create_graph("TargetActor")
    
    def set_training(self, args): 
        self.get_assign_target()
        self.get_bc_optimizers(args.bc_lr)
        self.get_rl_optimizers(args.rl_lr_a)


    # NN Architecture
    def create_graph(self, scope):
        initializer = tf.truncated_normal_initializer(stddev=0.01)
        outputs = {}

        with tf.variable_scope(scope):    
            with tf.variable_scope("Share"):
                fc1 = tf.layers.dense(self.feat, units=512, activation=tf.nn.relu, kernel_initializer=initializer)
                drop1 = tf.nn.dropout(fc1, self.dropout[0])

            for i in self.branch_list:
                with tf.variable_scope("Branch_" + str(i)):
                    fci1 = tf.layers.dense(drop1, units=256, activation=tf.nn.relu, kernel_initializer=initializer)
                    dpi1 = tf.nn.dropout(fci1, self.dropout[0])

                    fci2 = tf.layers.dense(dpi1, units=128, activation=tf.nn.relu, kernel_initializer=initializer)
                    dpi2 = tf.nn.dropout(fci2, self.dropout[0])

                    outputs[i] = tf.layers.dense(dpi2, units=2, activation=tf.nn.tanh, kernel_initializer=initializer)

        return outputs
    

    # Assign Actor to Target Actor
    def get_assign_target(self):
        self.tau_ph  = tf.placeholder(dtype= tf.float32, shape= 1) # for target network

        param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor")
        param_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="TargetActor")

        self.assign_target = [
            tf.assign(t, (1 - self.tau_ph[0]) * t + self.tau_ph[0] * p) for t, p in zip(param_target, param)
        ]
    
    def run_assign_target(self, tau):
        """If tau = 1, then copy actor to target actor"""
        self.sess.run(
            self.assign_target,
            feed_dict={
                self.tau_ph : [tau]
            }
        )


    # BC optimizer
    def get_bc_optimizers(self, lr):
        self.bc_optimizers = {}
        self.label_ph = tf.placeholder(dtype=tf.float32, shape=(None,2)) # for BC

        for i in self.branch_list:
            params = []
            params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Preprocessor") 
            params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/Share")
            params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/Branch_" + str(i))

            output = self.outputs[i]
            x = self.label_ph - output
            loss = tf.reduce_mean(x[:, 0] * x[:, 0] * 2 + x[:, 1] * x[:, 1]) # steer * 2 + velocity 

            self.bc_optimizers[i] = [
                loss, output, tf.train.AdamOptimizer(lr).minimize(loss, var_list=params)
            ]

    def run_bc_optimizers(self, images, lidars, measures, labels, branch, dropout):
        loss, out, _ = self.sess.run(
            self.bc_optimizers[branch],
            feed_dict = {
                self.image_ph: images,
                self.lidar_ph: lidars,
                self.measure_ph: measures,
                self.dropout: [dropout],
                self.label_ph: labels,
            }
        )
        return loss, out


    # RL optimizers
    def get_rl_optimizers(self, lr):
        self.rl_optimizers = {}
        self.gradient_ph = tf.placeholder(tf.float32, [None, 2]) # for RL

        for i in self.branch_list:
            params = []
            #params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Preprocessor")
            #params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/Share")
            params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/Branch_" + str(i))

            output = self.outputs[i]
            params_grad = tf.gradients(output, params, - self.gradient_ph)

            self.rl_optimizers[i] = tf.train.AdamOptimizer(lr).apply_gradients(zip(params_grad, params))

    def run_rl_optimizers(self, images, lidars, measures, gradient, branch, dropout):
        self.sess.run(
            self.rl_optimizers[branch],
            feed_dict={
                self.image_ph: images,
                self.lidar_ph: lidars,
                self.measure_ph: measures,
                self.dropout: [dropout],
                self.gradient_ph: gradient,
            }
        )


    # Prediction
    def predict_action(self, image, lidar, measure, branch, dropout):
        action = self.sess.run(
            self.outputs[branch],
            feed_dict={
                self.image_ph: image,
                self.lidar_ph: lidar,
                self.measure_ph: measure,
                self.dropout : [dropout],
            }
        )
        return action

    def predict_target_action(self, image, lidar, measure, branch, dropout):
        target_action = self.sess.run(
            self.target_outputs[branch],
            feed_dict={
                self.image_ph: image,
                self.lidar_ph: lidar,
                self.measure_ph: measure,
                self.dropout: [dropout]
            }
        )
        return target_action