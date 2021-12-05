import tensorflow as tf
import numpy as np

from collections import OrderedDict
from nn.layers import fc, droupout


class ActorNetwork(object):

    def __init__(self, sess, args, branch_list):
        self.sess = sess
        self.tau = args.tau
        self.lr_a = args.lr_a
        self.lr_bc = args.lr_bc
        self.hid_dim = args.hid_dim
        self.branch_list = branch_list

        self.image_ph,self.lidar_ph, self.measure_ph, self.branches = self.create_graph(scope="Actor")
        self.target_image_ph, self.target_lidar_ph, self.target_measure_ph,self.target_branches = self.create_graph(scope="TargetActor")
        
        self.action_gradient = tf.placeholder(tf.float32,[None, 2])
        self.target_optimize = self.get_target_optimize()

        self.ddpg_optimize = OrderedDict()
        for i in self.branch_list:
            self.ddpg_optimize[i] = self.get_ddpg_optimize(i)

        self.action_ph = tf.placeholder(dtype=tf.float32,shape=(None,2),name="ExpControl")

        self.bc_optimize = OrderedDict()
        for i in self.branch_list:
            self.bc_optimize[i] = self.get_bc_optimize(i)

        self.bc_target = self.get_bc_target()

    def create_graph(self,scope):
        branches = OrderedDict()

        with tf.variable_scope(scope):
            with tf.variable_scope("Share"):
                image_ph = tf.placeholder(dtype=tf.float32,shape=(None,512),name="ImageInput")
                lidar_ph = tf.placeholder(dtype=tf.float32,shape=(None,360),name="LidarInput")
                measure_ph = tf.placeholder(dtype=tf.float32,shape=(None,7),name="MeasureInput")

                lidar_fc1 = fc(lidar_ph, self.hid_dim, "lidar_layer_1", activation_fn='relu')
                measure_fc1 = fc(measure_ph, self.hid_dim, "measure_layer_1", activation_fn='relu')
                x_fc = tf.concat([image_ph, lidar_fc1, measure_fc1], axis=1)
                x_fc = fc(x_fc, 512, "concat_fc", activation_fn='relu')
                x_fc = droupout(x_fc, 0.2, "concat_fc") # TODO:

            for i in self.branch_list:
                scope_name = "branch_{}".format(i)
                with tf.name_scope(scope_name):
                    branch_output = fc(x_fc, self.hid_dim, scope_name+"_layer1", activation_fn='relu')
                    branch_output = droupout(branch_output, 0.2, scope_name+"_layer1")

                    branch_output = fc(branch_output, self.hid_dim, scope_name+"_layer2", activation_fn='relu')
                    branch_output = droupout(branch_output, 0.1, scope_name+"_layer2")

                    branch_output = fc(branch_output, 2, scope_name+"_out", activation_fn='tanh')
                branches[i] = branch_output

        return image_ph, lidar_ph, measure_ph, branches


    def get_weights(self,scope):
        all_weights =[]
        params_fc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
        all_weights.extend(params_fc)
        return all_weights


    def get_weights_branch(self,branch_num):
        branch_weights = []
        params_share = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Actor/Share")

        scope_name = "Actor/branch_{}".format(branch_num)
        params_branch = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope_name)
        branch_weights.extend(params_share)
        branch_weights.extend(params_branch)
        return branch_weights


    def get_target_optimize(self):
        params = self.get_weights(scope="Actor") 
        target_params = self.get_weights(scope="TargetActor")
        target_optimize=[tf.assign(t,(1-self.tau)*t+self.tau*e) for t,e in zip(target_params,params)]
        return target_optimize


    def get_bc_target(self):
        params = self.get_weights(scope="Actor")
        target_params = self.get_weights(scope="TargetActor")
        target_optimize = [tf.assign(t, e) for t, e in zip(target_params, params)]
        
        return target_optimize
        
    def run_bc_target(self):
        self.sess.run(self.bc_target)


    def get_ddpg_optimize(self, branch_num):
        branch_out = self.branches[branch_num]
        branch_params =self.get_weights_branch(branch_num)
        params_grad = tf.gradients(branch_out, branch_params, self.action_gradient)

        grads = zip(params_grad, branch_params)
        branch_optimize = tf.train.AdamOptimizer(-self.lr_a).apply_gradients(grads) #TODO: check
        return branch_optimize


    def get_bc_optimize(self, branch_num):
        branch_out = self.branches[branch_num]
        branch_params =self.get_weights_branch(branch_num)
        loss = tf.reduce_mean(tf.square(self.action_ph -  branch_out))
        return [loss, branch_out, tf.train.AdamOptimizer(self.lr_bc).minimize(loss, var_list=branch_params)]


    def train_bc(self, exp_images, exp_lidars, exp_measures, exp_actions, branch_num):
        loss, out, _ = self.sess.run(self.bc_optimize[branch_num], feed_dict={
            self.image_ph: exp_images,
            self.lidar_ph: exp_lidars,
            self.measure_ph: exp_measures,
            self.action_ph: exp_actions
        })
        return loss, out
    

    def train_target(self):
        self.sess.run(self.target_optimize)
    

    def train_ddpg(self, image_ph, lidar_ph, measure_ph, action_grads,branch_num):
        self.sess.run(self.ddpg_optimize[branch_num],feed_dict={
            self.image_ph:image_ph,
            self.lidar_ph:lidar_ph,
            self.measure_ph: measure_ph,
            self.action_gradient:action_grads
        })

    
    def predict_action(self, image_ph, lidar_ph, measure_ph, branch_num):
        """
        Arguments:
            branch_num {int}
        
        Returns:
            action (3,)
        """
        action = self.sess.run(self.branches[branch_num], feed_dict={
            self.image_ph:np.reshape(image_ph,(1, 512)),
            self.lidar_ph: np.reshape(lidar_ph, (1, 360)),
            self.measure_ph:np.reshape(measure_ph,(1, 7))
        })
        return action[0]


    def predict_target_action(self, target_image_ph, target_lidar_ph, target_measure_ph, branch_num):
        return self.sess.run(self.target_branches[branch_num],feed_dict={
            self.target_image_ph:target_image_ph,
            self.target_lidar_ph: target_lidar_ph,
            self.target_measure_ph:target_measure_ph
        })
