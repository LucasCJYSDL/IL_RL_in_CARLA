import h5py
import numpy as np
import time
from collections import OrderedDict
from dataset.image_augmentation import seq
from tools.others import render_image


import cv2

class Dset(object):
    def __init__(self, labels_0, labels_1, labels_2, labels_3, randomize, image_agent):

        self.labels_0 = labels_0
        self.labels_1 = labels_1
        self.labels_2 = labels_2
        self.labels_3 = labels_3

        self.randomize = randomize
        self.image_agent = image_agent
        self.num_pairs = len(labels_0)
        assert len(labels_1) == self.num_pairs
        assert len(labels_2) == self.num_pairs
        assert len(labels_3) == self.num_pairs

        print('# Get data:', self.num_pairs)

        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            #print("idx: ", idx)
            #print(self.labels_0.shape)
            self.labels_0 = self.labels_0[idx, :]
            self.labels_1 = self.labels_1[idx, :]
            self.labels_2 = self.labels_2[idx, :]
            self.labels_3 = self.labels_3[idx, :]

    def get_next_batch(self, batch_size):
        assert batch_size > 0

        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size

        labels_0 = self.labels_0[self.pointer:end, :]

        # render_image('2', labels_0[0])
        # cv2.waitKey(0)

        assert len(labels_0.shape) == 4, "False shape"
        assert labels_0.dtype == 'uint8', "False type"
        images_aug = seq(images=labels_0)

        # render_image('1', images_aug[0])
        # cv2.waitKey(0)

        # images_aug = labels_0
        images_aug = np.multiply(images_aug, 1.0 / 255.0)
        if self.image_agent:
            images_aug = self.image_agent.extract_feature(images_aug)
        labels_0 = images_aug
        labels_1 = self.labels_1[self.pointer:end, :]
        labels_2 = self.labels_2[self.pointer:end, :]
        labels_3 = self.labels_3[self.pointer:end, :]

        self.pointer = end

        return labels_0, labels_1, labels_2, labels_3

class Data_iters(object):

    def __init__(self, path, scene_list, item_list, branch_list, episode_ratio=1.0, is_shuffle=True, image_agent=None):
        self.path = path
        self.scene_list = scene_list
        self.item_list = item_list
        self.branch_list = branch_list
        self.episode_ratio = episode_ratio
        self.is_shuffle = is_shuffle
        self.image_agent = image_agent
        self.get_dataset()
        self.get_iters()
        del self.exp_dataset

    def get_dataset(self):

        self.exp_dataset = OrderedDict()
        for i in self.branch_list:
            self.exp_dataset[i] = {}
        for key in self.item_list:
            for i in self.branch_list:
                self.exp_dataset[i][key] = []
        #print("initial exp_data: ", self.exp_dataset)
        for scene_id in self.scene_list:
            file_name = self.path + "scene_" + str(scene_id) + ".h5"
            #print("processing: ", file_name)
            with h5py.File(file_name, 'r') as f:
                episode_num = int(self.episode_ratio * len(f.keys()))
                cnt = 0
                for key in f.keys():
                    #print(f[key])
                    for sub_key in self.item_list:
                        assert sub_key in f[key].keys()
                        #print(f[key][sub_key])
                        sub_value = f[key][sub_key][()]
                        branch = int(f[key]['task_type'][()][0])
                        #print("branch: ", branch)
                        self.exp_dataset[branch][sub_key].extend(sub_value)
                    cnt += 1
                    if cnt >= episode_num:
                        break

    def get_iters(self):

        self.dst_iters = OrderedDict()
        for ind in self.branch_list:
            temp_iters = Dset(np.array(self.exp_dataset[ind]['FrontRGB']), np.array(self.exp_dataset[ind]['Lidar']),
                              np.array(self.exp_dataset[ind]['Measurement']), np.array(self.exp_dataset[ind]['control']), self.is_shuffle, self.image_agent)
            self.dst_iters[ind] = temp_iters

    def get_next_branch(self, batchsize, branch):

        return self.dst_iters[branch].get_next_batch(batchsize)


if __name__ == "__main__":
    path = "../dataset/"
    start_time = time.time()
    from nn.image_agent import ImageAgent
    import tensorflow as tf
    import argparse

    argparser = argparse.ArgumentParser(description="DDPG for CARLA-0.9.7")
    argparser.add_argument('--lr_d', default=0.0001, type=float, help='learning rate for discriminator network')
    argparser.add_argument('--entcoeff', default=0.001, type=float, help='the coefficient for the entropy loss')
    argparser.add_argument('--hid_dim', default=32, type=int, help='the hidden dimension of a & c & d network')
    args = argparser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        img_agent = ImageAgent(sess)
        img_agent.load_model()
        test_iters = Data_iters(path, [0], ['FrontRGB', 'Lidar', 'Measurement', 'control'], episode_ratio=1.0, image_agent=img_agent, branch_list=[-1, 0, 1])
        end_time = time.time()
        print(end_time-start_time)

        for _ in range(2):
            exp_image, exp_lidar, exp_measure, exp_action = test_iters.get_next_branch(1, 0)
            print(exp_image.shape)
            print(exp_lidar.shape, " ", exp_lidar)
            print(exp_measure.shape, " ", exp_measure)
            print(exp_action.shape, " ", exp_action)
            end_sample_time = time.time()
            print(end_sample_time - end_time)
            end_time = end_sample_time

        # from nn.DiscriminatorNetwork import Discriminator
        # discriminator = Discriminator(sess, args, [-1, 0, 1])
        #
        # sess.run(tf.global_variables_initializer())
        # sess.graph.finalize()
        #
        # # train the discriminator and get the reward
        # exp_image, exp_speed, exp_action = test_iters.get_next_branch(64, 0)
        # states_image = np.zeros_like(exp_image)
        # states_speed = np.zeros_like(exp_speed)
        # actions = np.zeros_like(exp_action)
        # d_loss = discriminator.train_branch(states_image, states_speed, actions, exp_image, exp_speed, exp_action, 0)
        # rewards = discriminator.predict_reward(states_image, states_speed, actions, 0)
        # print("rewards shape: ", rewards.shape, rewards)


