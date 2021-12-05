import random
import numpy as np
import tensorflow as tf

from collections import OrderedDict

from nn.ActorNetwork import ActorNetwork
from nn.CriticNetwork import CriticNetwork
from nn.DiscriminatorNetwork import Discriminator


class Learner:

    def __init__(self, sess, args, branch_list):
        print('# Building Learner')

        self.sess = sess
        self.branch_list = branch_list

        self.actor = ActorNetwork(sess, args, branch_list)
        self.critic = CriticNetwork(sess, args, branch_list)
        self.discriminator = Discriminator(sess, args, branch_list)

        self.bc_batch = args.bc_batch
        self.rl_batch = args.rl_batch
        self.bc_update = args.bc_update

        self.lam = args.lam
        self.gamma = args.gamma
        self.ddpg_update = args.ddpg_update
        self.discriminator_update = args.discriminator_update
    
    def assgin_exp_dataset(self, exp_dataset):
        self.exp_dataset = exp_dataset

    def train_bc(self):
        loss_list = {}

        print('train bc batch', self.bc_update)
        for branch in self.branch_list:
            loss_list[branch] = []
        
        for _ in range(self.bc_update):
            for branch in self.branch_list:
                exp_images, exp_lidars, exp_measures, raw_exp_actions = self.exp_dataset.get_next_branch(self.bc_batch, branch)
                batch_size = len(exp_images)

                assert exp_images.shape == (batch_size, 512)
                assert exp_lidars.shape == (batch_size, 360)
                assert exp_measures.shape == (batch_size, 7)
                assert raw_exp_actions.shape == (batch_size, 3)

                exp_actions = np.zeros([batch_size, 2])
                exp_actions[:, 0] += raw_exp_actions[:, 0]
                exp_actions[:, 1] = raw_exp_actions[:, 1] - raw_exp_actions[:, 2]
                assert exp_actions.shape == (batch_size, 2)
                # print("before: ", exp_measures)
                exp_measures[:,0] = exp_measures[:, 0]/30. #TODO:
                exp_measures[:,1] = exp_measures[:, 1]/25.
                exp_measures[:,2] = exp_measures[:, 2]/25.
                # print("after: ", exp_measures)
                loss, __ = self.actor.train_bc(exp_images, exp_lidars, exp_measures, exp_actions, branch)
                loss_list[branch].append(loss)

                '''
                if branch == 1 and _ == self.bc_update - 1:
                    print('branch:', branch)
                    print('expect:\n', np.array(exp_actions))
                    print('out:\n', np.array(out_action))
                '''
                
        
        self.actor.run_bc_target()

        bc_losses = OrderedDict()
        for branch in self.branch_list:
            if branch in loss_list:
                bc_losses[str(branch)] = np.mean(loss_list[branch])
        
        return bc_losses


    def train_ddpg(self, buffer_dict):
        print('# Training DDPG', self.ddpg_update)

        loss_list = {}
        for branch in self.branch_list:
            if buffer_dict.count(branch) < self.rl_batch:
                continue
            loss_list[branch] = []

        for _ in range(self.ddpg_update):
            for branch in self.branch_list:
                if buffer_dict.count(branch) < self.rl_batch:
                    continue

                batch = buffer_dict.getBatch(branch, self.rl_batch)

                st_images = np.asarray([e[0][0] for e in batch]) # (None, 512)
                st_lidars = np.asarray([e[0][1] for e in batch])
                st_measures = np.asarray([e[0][2] for e in batch])
                actions = np.asarray([e[1] for e in batch]) # (None, 2)

                rewards_1 = np.asarray([e[2] for e in batch]) # (None, 1)
                rewards_2 = self.discriminator.predict_reward(st_images, st_lidars, st_measures, actions, branch)
                assert rewards_2.shape == (actions.shape[0], 1)

                rewards = rewards_1 #+ rewards_2 * self.lam

                st1_images = np.asarray([e[3][0] for e in batch]) # (None, 512)
                st1_lidars = np.asarray([e[3][1] for e in batch])
                st1_measures = np.asarray([e[3][2] for e in batch])
                dones = np.asarray([e[4] for e in  batch]) # (None, 1)

                target_action = self.actor.predict_target_action(st1_images, st1_lidars, st1_measures, branch)
                target_q = self.critic.predict_target_q(st1_images, st1_lidars, st1_measures, target_action, branch)
                advance = rewards + (1 - dones) * target_q * self.gamma  #TODO: Normalize advance.
                
                loss = self.critic.train_branch(st_images, st_lidars, st_measures, actions, advance, branch)
                loss_list[branch].append(loss)

                predict_action = self.actor.predict_target_action(st_images, st_lidars, st_measures, branch)
                action_gradient = self.critic.run_gradient(st_images, st_lidars, st_measures, predict_action, branch)
                
                self.actor.train_ddpg(st_images, st_lidars, st_measures, action_gradient, branch)
                self.actor.train_target()
                self.critic.train_target()

                '''
                assert st_images.shape == (batch_size, 512)
                assert st_speeds.shape == (batch_size, 1)
                assert actions.shape == (batch_size, 3)
                assert rewards.shape == (batch_size, 1)
                assert st1_images.shape == (batch_size, 512)
                assert st1_speeds.shape == (batch_size, 1)
                assert dones.shape == (batch_size, 1)
                assert target_action.shape == (batch_size, 3)
                assert target_q.shape == (batch_size, 1)
                assert advance.shape == (batch_size, 1)
                assert predict_action.shape == (batch_size, 3)
                assert action_gradient.shape == (batch_size, 3)
                '''
        
        ddpg_losses = OrderedDict()
        for branch in self.branch_list:
            if branch in loss_list:
                ddpg_losses[str(branch)] = np.mean(loss_list[branch])
        
        return ddpg_losses


    def train_discriminator(self, buffer_dict):
        print('# Training Discriminator')

        loss_list = {}
        for branch in self.branch_list:
            if buffer_dict.count(branch) < self.rl_batch:
                continue
            loss_list[branch] = []

        for _ in range(self.discriminator_update):
            for branch in self.branch_list:
                if buffer_dict.count(branch) < self.rl_batch:
                    continue

                batch = buffer_dict.getBatch(branch, self.rl_batch)

                st_images = np.asarray([e[0][0] for e in batch])  # (None, 512)
                st_lidars = np.asarray([e[0][1] for e in batch])
                st_measures = np.asarray([e[0][2] for e in batch])

                actions = np.asarray([e[1] for e in batch])  # (None, 2)

                exp_images, exp_lidars, exp_measures, raw_exp_actions = self.exp_dataset.get_next_branch(self.rl_batch, branch)
                batch_size = len(exp_images)
                exp_actions = np.zeros([batch_size, 2])
                exp_actions[:, 0] += raw_exp_actions[:, 0]
                exp_actions[:, 1] = raw_exp_actions[:, 1] - raw_exp_actions[:, 2]
                assert exp_actions.shape == (batch_size, 2)
                exp_measures[:, 0] = exp_measures[:, 0] / 30.  # TODO:
                exp_measures[:, 1] = exp_measures[:, 1] / 25.
                exp_measures[:, 2] = exp_measures[:, 2] / 25.
                loss = self.discriminator.train_branch(st_images, st_lidars, st_measures, actions, exp_images, exp_lidars, exp_measures, exp_actions, branch)
                loss_list[branch].append(loss)
        
        discriminator_losses = OrderedDict()
        for branch in self.branch_list:
            if branch in loss_list:
                discriminator_losses[str(branch)] = np.mean(loss_list[branch])
        
        return discriminator_losses


