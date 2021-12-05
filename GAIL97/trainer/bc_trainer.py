'''
BC Trainer: Behavior Cloning
'''


import numpy as np
import tensorflow as tf
from collections import OrderedDict

from tools.others import render_image, ImgAug


class BCTrainer:

    def __init__(self, args, actor, exp_data, branch_list):
        self.actor = actor
        self.exp_data = exp_data
        self.branch_list = branch_list

        self.bc_batch = args.bc_batch
        self.bc_update = args.bc_update
        
        self.bc_dropout = args.bc_dropout


    def train(self, debug=False):
        print('# update = %d, batch size = %d' % (self.bc_update, self.bc_batch))
        #img_aug = ImgAug() #TODO:

        loss_list = OrderedDict()
        for branch in self.branch_list:
            loss_list[branch] = []

        for _ in range(self.bc_update):
            for branch in self.branch_list:
                samples = self.exp_data.get_batch(branch, self.bc_batch)

                img_t = samples['img_t'] / 255
                lid_t = samples['lid_t']
                mea_t = samples['mea_t']
                a_t = samples['a_t']

                loss, __ = self.actor.run_bc_optimizers(img_t, lid_t, mea_t, a_t, branch, dropout=self.bc_dropout)
                loss_list[branch].append(loss)

                if debug:
                    render_image('111', img_t[0])
                    print(img_t[0], lid_t[0], mea_t[0])
        
        mean_loss = []
        for (k, v) in loss_list.items():
            mean_loss += v
            print('# branch = %d, loss = %lf' % (k, np.mean(v)))
        
        print('# mean loss:', np.mean(mean_loss))
        return np.mean(mean_loss)