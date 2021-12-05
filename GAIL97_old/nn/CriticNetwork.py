import tensorflow as tf
import numpy as np

from collections import OrderedDict
from nn.layers import fc, droupout


class CriticNetwork(object):

    def __init__(self, sess, args, branch_list):
        self.sess = sess
        self.tau = args.tau
        self.lr_c = args.lr_c
        self.hid_dim = args.hid_dim
        self.branch_list = branch_list

        self.image_ph, self.lidar_ph, self.measure_ph, self.action_ph, self.branches = self.create_graph(scope="Critic")
        self.target_image_ph,self.target_lidar_ph, self.target_measure_ph,self.target_action,self.target_branches = self.create_graph(scope="TargetCritic")
        
        self.target_q = tf.placeholder(tf.float32,[None, 1])
        self.target_optimize = self.get_target_optimize()

        self.branch_optimizes = OrderedDict()
        for i in self.branch_list:
            self.branch_optimizes[i] = self.get_branch_optimize(i)

        self.action_gradients = OrderedDict()
        for i in self.branch_list:
            self.action_gradients[i] = tf.gradients(self.branches[i], self.action_ph)


    def create_graph(self,scope):
        branches = OrderedDict()

        with tf.variable_scope(scope):
            with tf.variable_scope("Share"):
                image_ph = tf.placeholder(dtype=tf.float32, shape=(None,512),name="ImageInput")
                lidar_ph = tf.placeholder(dtype=tf.float32, shape=(None, 360), name="LidarInput")
                measure_ph = tf.placeholder(dtype=tf.float32, shape=(None, 7), name="MeasureInput")
                action_ph = tf.placeholder(dtype=tf.float32, shape=(None,2),name="ActionInput")
                lidar_fc1 = fc(lidar_ph, self.hid_dim, "lidar_layer_1", activation_fn='relu')
                measure_fc1 = fc(measure_ph, self.hid_dim, "measure_layer_1", activation_fn='relu')
                action_fc1 = fc(action_ph, self.hid_dim, "action_layer_1", activation_fn='relu')

                x_fc = tf.concat([image_ph, lidar_fc1, measure_fc1, action_fc1], axis=1)
                x_fc = fc(x_fc, 512, "concat_fc", activation_fn='relu')

            for i in self.branch_list:
                scope_name = "branch_{}".format(i)
                with tf.name_scope(scope_name):
                    branch_output = fc(x_fc, self.hid_dim, scope_name+"_layer1", activation_fn='relu')
                    branch_output = fc(branch_output, self.hid_dim, scope_name+"_layer2", activation_fn='relu')
                    branch_output = fc(branch_output, 1, scope_name+"_out", activation_fn='tanh')
                branches[i] = branch_output

        return image_ph, lidar_ph, measure_ph, action_ph, branches


    def get_weights(self,scope):
        all_weights =[]
        params_fc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
        all_weights.extend(params_fc)
        return all_weights
    

    def get_weights_branch(self,branch_num):
        branch_weights = []
        params_share = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Critic/Share")
        
        scope_name = "Critic/branch_{}".format(branch_num)
        params_branch = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope_name)
        branch_weights.extend(params_share)
        branch_weights.extend(params_branch)
        return branch_weights


    def get_target_optimize(self):
        params = self.get_weights(scope="Critic") 
        target_params = self.get_weights(scope="TargetCritic")
        target_optimize=[tf.assign(t, (1-self.tau)*t+self.tau*e) for t,e in zip(target_params,params)]
        return target_optimize


    def get_branch_optimize(self, branch_num):
        branch_out = self.branches[branch_num]
        branch_params =self.get_weights_branch(branch_num)
        loss = tf.reduce_mean(tf.squared_difference(self.target_q, branch_out))
        return [loss, tf.train.AdamOptimizer(self.lr_c).minimize(loss, var_list=branch_params)]


    def train_target(self):
        self.sess.run(self.target_optimize)

    
    def run_gradient(self,image,lidar,measure,action,branch_num):
        return self.sess.run(self.action_gradients[branch_num],feed_dict={
            self.image_ph:image,
            self.lidar_ph: lidar,
            self.measure_ph: measure,
            self.action_ph:action
        })[0]
    
    
    def train_branch(self, image_ph, lidar_ph, measure_ph, action_ph, target_q, branch_num):
        loss, _ = self.sess.run(self.branch_optimizes[branch_num],feed_dict={
            self.image_ph:image_ph,
            self.lidar_ph: lidar_ph,
            self.measure_ph: measure_ph,
            self.action_ph:action_ph,
            self.target_q:target_q,
        })
        return loss


    def predict_target_q(self, target_image_ph, target_lidar_ph, target_measure_ph, target_action, branch_num):
        return self.sess.run(self.target_branches[branch_num],feed_dict={
            self.target_image_ph:target_image_ph,
            self.target_lidar_ph: target_lidar_ph,
            self.target_measure_ph: target_measure_ph,
            self.target_action:target_action
        })

