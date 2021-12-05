'''
RL Trainer: DDPG
'''


import numpy as np
import tensorflow as tf
from collections import OrderedDict

from tools.others import render_image, ImgAug


class RLTrainer:

    def __init__(self, args, actor, critic, buffers, branch_list):
        self.actor = actor
        self.critic = critic
        self.buffers = buffers
        self.branch_list = branch_list

        self.rl_batch = args.rl_batch
        self.rl_update = args.rl_update

        self.rl_dropout = args.rl_dropout
        self.rl_gamma = args.rl_gamma
        self.rl_tau = args.rl_tau


    def train(self, train_actor, debug=False):
        print('# update = %d, batch size = %d, train_actor = %d' % (self.rl_update, self.rl_batch, train_actor))

        loss_list = OrderedDict()
        for branch in self.branch_list:
            loss_list[branch] = []

        for _ in range(self.rl_update):
            for branch in self.branch_list:
                samples = self.buffers.get_batch(branch, self.rl_batch)

                img_t = samples['img_t'] / 255
                lid_t = samples['lid_t']
                mea_t = samples['mea_t']

                a_t = samples['a_t']
                r_t = samples['r_t']

                img_t1 = samples['img_t1'] / 255
                lid_t1 = samples['lid_t1']
                mea_t1 = samples['mea_t1']

                done = samples['done']

                assert r_t.shape == (self.rl_batch, 5), r_t.shape
                assert a_t.shape == (self.rl_batch, 2), a_t.shape

                r_t_value = r_t @ np.array([-0.5, -0.1, -0.1, -0.3, 0.2])
                r_t_value += (a_t > 0.9) @ np.array([0.0, -0.1])
                r_t_value += (a_t > 0.3) @ np.array([0.0, 0.1])

                assert r_t_value.shape == (self.rl_batch, ), r_t_value.shape
                r_t_value = np.reshape(r_t_value, (self.rl_batch, 1))

                # train critic
                a_t1 = self.actor.predict_target_action(img_t1, lid_t1, mea_t1, branch, dropout=self.rl_dropout)
                q_t1 = self.critic.predict_target_q(img_t1, lid_t1, mea_t1, a_t1, branch, dropout=self.rl_dropout)

                label_q = r_t_value + (1 - done) * q_t1 * self.rl_gamma

                assert a_t1.shape == (self.rl_batch, 2), a_t1.shape
                assert q_t1.shape == (self.rl_batch, 1), q_t1.shape

                loss, out = self.critic.run_rl_optimizers(img_t, lid_t, mea_t, a_t, label_q, branch, dropout=self.rl_dropout)
                loss_list[branch].append(loss)

                self.critic.run_assign_target(self.rl_tau)

                if branch == 0 and _ == 0:
                    print('r_t_value', r_t_value[:10])
                    print('label_q:', label_q[:10])
                    print('predict_qt', out[:10])
                    print('loss:', loss)

                # train actor
                if not train_actor:
                    continue

                predict_action = self.actor.predict_action(img_t, lid_t, mea_t, branch, dropout=self.rl_dropout)
                action_gradient = self.critic.run_action_gradients(img_t, lid_t, mea_t, predict_action, branch, dropout=self.rl_dropout)

                assert predict_action.shape == (self.rl_batch, 2), predict_action.shape
                assert action_gradient.shape == (self.rl_batch, 2), action_gradient.shape

                self.actor.run_rl_optimizers(img_t, lid_t, mea_t, action_gradient, branch, dropout=self.rl_dropout)
                self.actor.run_assign_target(self.rl_tau)
        
        mean_loss = []
        for (k, v) in loss_list.items():
            mean_loss += v
            print('# branch = %d, loss = %lf' % (k, np.mean(v)))
        
        print('# mean loss:', np.mean(mean_loss))
        return np.mean(mean_loss)

