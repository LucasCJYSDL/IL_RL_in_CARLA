'''
RL Trainer: DDPG
'''

import random
import numpy as np
import tensorflow as tf

from collections import OrderedDict


class RLTrainer:

    def __init__(self, args, actor, critic, replay, branch_list):
        self.actor = actor
        self.critic = critic
        self.replay = replay
        self.branch_list = branch_list

        self.rl_batch = args.rl_batch
        self.rl_update = args.rl_update

        self.rl_dropout = args.rl_dropout
        self.rl_gamma = args.rl_gamma
        self.rl_tau = args.rl_tau


    def train(self):
        print('# update = %d, batch size = %d' % (self.rl_update, self.rl_batch))

        rwd_list, rl_loss_list = [], []

        for _ in range(self.rl_update):
            for branch in self.branch_list:
                if self.rl_batch * (_ + 1) > self.replay.count(branch):
                    continue

                #if branch != -1:
                #    continue

                samples = self.replay.get_batch(branch, self.rl_batch)

                img_t = samples['img_t'] / 255
                lid_t = samples['lid_t']
                mea_t = samples['mea_t']

                a_t = samples['a_t']
                r_t = samples['r_t']

                img_t1 = samples['img_t1'] / 255
                lid_t1 = samples['lid_t1']
                mea_t1 = samples['mea_t1']

                done = samples['done']

                assert r_t.shape == (self.rl_batch, 3), r_t.shape
                assert a_t.shape == (self.rl_batch, 2), a_t.shape

                r_t_value = r_t @ np.array([-0.5, -0.1, 0.0])           # collision, lane, success
                r_t_value += - abs(mea_t1[:, 1]) / 4.0
                r_t_value += (a_t[:, 1] > 0.3) * 0.1

                #r_t_value += (mea_t[:, 0] > 1.0) * -0.1                  # exceed speed
                #r_t_value += (abs(a_t) > 0.9) @ np.array([0.0, -0.4])   # jerk

                assert r_t_value.shape == (self.rl_batch, ), r_t_value.shape
                r_t_value = np.reshape(r_t_value, (self.rl_batch, 1))
                rwd_list.append(np.mean(r_t_value))

                if random.randint(0, 10) == 0:
                    print('r_t_value', r_t_value[:5])

                ##### train critic
                a_t1 = self.actor.predict_target_action(img_t1, lid_t1, mea_t1, branch, dropout=self.rl_dropout)
                q_t1 = self.critic.predict_target_q(img_t1, lid_t1, mea_t1, a_t1, branch, dropout=self.rl_dropout)

                label_q = r_t_value + (1 - done) * q_t1 * self.rl_gamma

                assert a_t1.shape == (self.rl_batch, 2), a_t1.shape
                assert q_t1.shape == (self.rl_batch, 1), q_t1.shape

                rl_loss, out = self.critic.run_rl_optimizers(img_t, lid_t, mea_t, a_t, label_q, branch, dropout=self.rl_dropout)
                rl_loss_list.append(rl_loss)

                self.critic.run_assign_target(self.rl_tau)
                
                ###### train actor
                predict_action = self.actor.predict_action(img_t, lid_t, mea_t, branch, dropout=self.rl_dropout)
                action_gradient = self.critic.run_action_gradients(img_t, lid_t, mea_t, predict_action, branch, dropout=self.rl_dropout)

                assert predict_action.shape == (self.rl_batch, 2), predict_action.shape
                assert action_gradient.shape == (self.rl_batch, 2), action_gradient.shape

                self.actor.run_rl_optimizers(img_t, lid_t, mea_t, action_gradient, branch, dropout=self.rl_dropout)
                self.actor.run_assign_target(self.rl_tau)
        
        
        mean_rwd = np.mean(rwd_list)
        mean_rl_loss = np.mean(rl_loss_list)

        print('# mean reward', mean_rwd)
        print('# mean rl_loss', mean_rl_loss)

        return mean_rwd, mean_rl_loss

