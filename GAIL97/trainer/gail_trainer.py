"""
GAIL Trainer
"""

import random
import numpy as np
import tensorflow as tf
from collections import OrderedDict


class GAILTrainer:

    def __init__(self, args, actor, critic, discriminator, exp_data, replay, branch_list):
        self.actor = actor
        self.critic = critic
        self.discriminator = discriminator

        self.exp_data = exp_data
        self.replay = replay
        self.branch_list = branch_list

        self.gail_update = args.gail_update
        self.gail_batch = args.gail_batch
        self.gail_G = args.gail_G
        
        self.rl_dropout = args.rl_dropout
        self.rl_gamma = args.rl_gamma
        self.rl_tau = args.rl_tau


    def update_rl(self, branch):
        samples = self.replay.get_batch(branch, self.gail_batch)

        img_t = samples['img_t'] / 255
        lid_t = samples['lid_t']
        mea_t = samples['mea_t']

        a_t = samples['a_t']
        r_t = samples['r_t']

        img_t1 = samples['img_t1'] / 255
        lid_t1 = samples['lid_t1']
        mea_t1 = samples['mea_t1']

        done = samples['done']

        assert r_t.shape == (self.gail_batch, 3), r_t.shape
        assert a_t.shape == (self.gail_batch, 2), a_t.shape

        r_t_value = self.discriminator.predict_reward(img_t, lid_t, mea_t, a_t, branch, dropout=self.rl_dropout)
        assert r_t_value.shape == (self.gail_batch, 1), r_t_value.shape

        r_t_value += (r_t @ np.array([-0.5, -0.1, 0])).reshape((self.gail_batch, 1))
        assert r_t_value.shape == (self.gail_batch, 1), r_t_value.shape

        if random.randint(0, 10) == 0:
            print('r_t', r_t_value[:5])

        ##### train critic
        a_t1 = self.actor.predict_target_action(img_t1, lid_t1, mea_t1, branch, dropout=self.rl_dropout)
        q_t1 = self.critic.predict_target_q(img_t1, lid_t1, mea_t1, a_t1, branch, dropout=self.rl_dropout)
        label_q = r_t_value + (1 - done) * q_t1 * self.rl_gamma

        assert a_t1.shape == (self.gail_batch, 2), a_t1.shape
        assert q_t1.shape == (self.gail_batch, 1), q_t1.shape

        rl_loss, out = self.critic.run_rl_optimizers(img_t, lid_t, mea_t, a_t, label_q, branch, dropout=self.rl_dropout)
        self.critic.run_assign_target(self.rl_tau)
        
        ###### train actor
        predict_action = self.actor.predict_action(img_t, lid_t, mea_t, branch, dropout=self.rl_dropout)
        action_gradient = self.critic.run_action_gradients(img_t, lid_t, mea_t, predict_action, branch, dropout=self.rl_dropout)

        assert predict_action.shape == (self.gail_batch, 2), predict_action.shape
        assert action_gradient.shape == (self.gail_batch, 2), action_gradient.shape

        self.actor.run_rl_optimizers(img_t, lid_t, mea_t, action_gradient, branch, dropout=self.rl_dropout)
        self.actor.run_assign_target(self.rl_tau)

        return rl_loss, np.mean(r_t_value)


    def update_disc(self, branch):
        g_samples = self.replay.get_batch(branch, self.gail_batch)
        e_samples = self.exp_data.get_batch(branch, self.gail_batch)

        g_img = g_samples['img_t'] / 255
        g_lid = g_samples['lid_t']
        g_mea = g_samples['mea_t']
        g_act = g_samples['a_t']

        e_img = e_samples['img_t'] / 255
        e_lid = e_samples['lid_t']
        e_mea = e_samples['mea_t']
        e_act = e_samples['a_t']

        d_loss, g_acc, e_acc = self.discriminator.run_gail_optimizers(g_img, g_lid, g_mea, g_act, e_img, e_lid, e_mea, e_act, branch, dropout=self.rl_dropout)
        return d_loss, g_acc, e_acc
    

    def train(self):
        print('# train GAIL, update = %d, batch size = %d' % (self.gail_update, self.gail_batch))

        rwd_list, rl_loss_list, d_loss_list, g_acc_list, e_acc_list = [], [], [], [], []

        for _ in range(self.gail_update):
            for branch in self.branch_list:
                if self.gail_batch * (_ + 1) > self.replay.count(branch):
                    continue
                
                #if branch != 1:
                #    continue
                
                for __ in range(self.gail_G):
                    rl_loss, rwd = self.update_rl(branch)
                d_loss, g_acc, e_acc = self.update_disc(branch)

                rwd_list.append(rwd)
                rl_loss_list.append(rl_loss)
                d_loss_list.append(d_loss)
                g_acc_list.append(g_acc)
                e_acc_list.append(e_acc)
        
        mean_rwd = np.mean(rwd_list)
        mean_rl_loss = np.mean(rl_loss_list)
        mean_d_loss = np.mean(d_loss_list)
        mean_g_acc = np.mean(g_acc_list)
        mean_e_acc = np.mean(e_acc_list)

        print('# mean reward', mean_rwd)
        print('# mean rl_loss', mean_rl_loss)
        print('# mean d_loss', mean_d_loss)
        print('# mean g_acc', mean_g_acc)
        print('# mean e_acc', mean_e_acc)

        return mean_rwd, mean_rl_loss, mean_d_loss, mean_g_acc, mean_e_acc