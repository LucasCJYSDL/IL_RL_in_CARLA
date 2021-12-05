"""
The discriminator layer, map (s_t, a_t) to a float (the prob. to be true)
"""

import numpy as np
import tensorflow as  tf


def logsigmoid(a):
    return -tf.nn.softplus(-a)

def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent



class Discriminator:

    def __init__(self, sess, preprocessor, branch_list):
        self.sess = sess
        self.branch_list = branch_list
        
        self.image_ph, self.lidar_ph, self.measure_ph, self.dropout = preprocessor.get_input()
        self.feat = preprocessor.get_feat()

        # generator
        self.g_feat_ph = tf.placeholder(dtype=tf.float32, shape=(None, 896))
        self.g_act_ph = tf.placeholder(dtype=tf.float32, shape=(None, 2))
        self.g_probs = self.create_graph('Discriminator', self.g_feat_ph, self.g_act_ph, reuse=False)

        # expert
        self.e_feat_ph = tf.placeholder(dtype=tf.float32, shape=(None, 896))
        self.e_act_ph = tf.placeholder(dtype=tf.float32, shape=(None, 2))
        self.e_probs = self.create_graph('Discriminator', self.e_feat_ph, self.e_act_ph, reuse=True)         # reuse


    def set_training(self, args):
        self.get_gail_optimizers(args.gail_lr_d, args.gail_entcoeff)


    def create_graph(self, scope, feat_ph, act_ph, reuse):
        initializer = tf.truncated_normal_initializer(stddev=0.01)
        probs = {}

        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope("Share"):
                fc0 = tf.layers.dense(act_ph, units=64, activation=tf.nn.tanh, kernel_initializer=initializer)
                cat1 = tf.concat([fc0, feat_ph], axis=1)

                #fc1 = tf.layers.dense(cat1, units=512, activation=tf.nn.tanh, kernel_initializer=initializer)
                #drop1 = tf.nn.dropout(fc1, self.dropout[0])

            for i in self.branch_list:
                with tf.variable_scope("Branch_" + str(i)):
                    fci1 = tf.layers.dense(cat1, units=256, activation=tf.nn.tanh, kernel_initializer=initializer)
                    dpi1 = tf.nn.dropout(fci1, self.dropout[0])

                    probs[i] = tf.layers.dense(dpi1, units=1, kernel_initializer=initializer) # shouldn't take activation

        return probs
    

    def get_gail_optimizers(self, lr, entcoeff):
        self.rwds = {}
        self.gail_optimizers = {}

        for i in self.branch_list:
            params = []
            #params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Preprocessor")
            params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator/Share")
            params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator/Branch_" + str(i))

            g_prob = self.g_probs[i]
            e_prob = self.e_probs[i]

            self.rwds[i] = - tf.log(1 - tf.nn.sigmoid(g_prob) + 1e-8)
            #self.rwds[i] = tf.log(tf.nn.sigmoid(g_prob) + 1e-8)

            g_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(g_prob) < 0.5))
            e_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(e_prob) > 0.5))

            # let x = logits, z = targets.
            # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=g_prob, labels=tf.zeros_like(g_prob))
            g_loss = tf.reduce_mean(g_loss)
        
            e_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=e_prob, labels=tf.ones_like(e_prob))
            e_loss = tf.reduce_mean(e_loss)

            logits = tf.concat([g_prob, e_prob], axis=0)
            entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
            entropy_loss = - entcoeff * entropy

            total_loss = g_loss + e_loss + entropy_loss

            self.gail_optimizers[i] = [
                total_loss, g_acc, e_acc, tf.train.AdamOptimizer(lr).minimize(total_loss, var_list=params)
            ]
        
    
    def run_gail_optimizers(self, g_img, g_lid, g_mea, g_act, e_img, e_lid, e_mea, e_act, branch, dropout):
        g_feat = self.sess.run(
            self.feat,
            feed_dict={
                self.image_ph: g_img,
                self.lidar_ph: g_lid,
                self.measure_ph: g_mea,
                self.dropout: [dropout],
            }
        )

        e_feat = self.sess.run(
            self.feat,
            feed_dict={
                self.image_ph: e_img,
                self.lidar_ph: e_lid,
                self.measure_ph: e_mea,
                self.dropout: [dropout],
            }
        )

        total_loss, g_acc, e_acc, _ = self.sess.run(
            self.gail_optimizers[branch],
            feed_dict={
                self.g_feat_ph : g_feat,
                self.g_act_ph : g_act,
                self.e_feat_ph : e_feat,
                self.e_act_ph : e_act,
                self.dropout: [dropout],
            }
        )

        return total_loss, g_acc, e_acc


    def predict_reward(self, g_img, g_lid, g_mea, g_act, branch, dropout):
        g_feat = self.sess.run(
            self.feat,
            feed_dict={
                self.image_ph: g_img,
                self.lidar_ph: g_lid,
                self.measure_ph: g_mea,
                self.dropout: [dropout],
            }
        )

        rwd = self.sess.run(
            self.rwds[branch],
            feed_dict={
                self.g_feat_ph : g_feat,
                self.g_act_ph : g_act,
                self.dropout: [dropout],

            }
        )
        return rwd