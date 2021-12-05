import h5py
import numpy as np
import os
import time

class Dset(object):
    def __init__(self, labels_0, labels_1, labels_2, labels_3, labels_4, randomize):

        self.labels_0 = labels_0
        self.labels_1 = labels_1
        self.labels_2 = labels_2
        self.labels_3 = labels_3
        self.labels_4 = labels_4
        # self.labels_5 = labels_5
        self.randomize = randomize
        self.num_pairs = len(labels_0)
        assert len(labels_1) == self.num_pairs
        assert len(labels_2) == self.num_pairs
        assert len(labels_3) == self.num_pairs
        assert len(labels_4) == self.num_pairs
        # assert len(labels_5) == self.num_pairs
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)

            self.labels_0 = self.labels_0[idx, :]
            self.labels_1 = self.labels_1[idx, :]
            self.labels_2 = self.labels_2[idx, :]
            self.labels_3 = self.labels_3[idx, :]
            self.labels_4 = self.labels_4[idx, :]
            # self.labels_5 = self.labels_5[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.labels_0, self.labels_1, self.labels_2, self.labels_3, self.labels_4
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size

        labels_0 = self.labels_0[self.pointer:end, :]
        labels_1 = self.labels_1[self.pointer:end, :]
        labels_2 = self.labels_2[self.pointer:end, :]
        labels_3 = self.labels_3[self.pointer:end, :]
        labels_4 = self.labels_4[self.pointer:end, :]
        # labels_5 = self.labels_5[self.pointer:end, :]

        self.pointer = end

        return labels_0, labels_1, labels_2, labels_3, labels_4#, labels_5

class Data_iters(object):

    def __init__(self, path, scene_list, item_list, episode_ratio=1.0, is_shuffle=True):
        self.path = path
        self.scene_list = scene_list
        self.item_list = item_list
        self.episode_ratio = episode_ratio
        self.is_shuffle = is_shuffle
        self.get_dataset()
        self.get_iters()
        del self.exp_dataset

    def get_dataset(self):

        self.exp_dataset = {}
        for key in self.item_list:
            self.exp_dataset[key] = []
        print("initial exp_data: ", self.exp_dataset)
        for scene_id in self.scene_list:
            file_name = self.path + "scene_" + str(scene_id) + ".h5"
            print("processing: ", file_name)
            with h5py.File(file_name, 'r') as f:
                episode_num = int(self.episode_ratio * len(f.keys()))
                cnt = 0
                for key in f.keys():
                    print(f[key])
                    for sub_key in self.item_list:
                        assert sub_key in f[key].keys()
                        print(f[key][sub_key])
                        self.exp_dataset[sub_key].extend(f[key][sub_key][()])

        self.exp_dataset['mean'] = self.exp_dataset['control']
        self.exp_dataset['logstd'] = np.zeros_like(self.exp_dataset['mean']) + np.log(1e-2)

    def get_iters(self):

        self.dst_iters = Dset(np.array(self.exp_dataset['FrontRGB']), np.array(self.exp_dataset['FrontSemantic']),
                              np.array(self.exp_dataset['speed']), np.array(self.exp_dataset['control']), np.array(self.exp_dataset['task_type']), self.is_shuffle)
                              # np.array(self.exp_dataset['mean']), np.array(self.exp_dataset['logstd']), self.is_shuffle)

    def get_next_branch(self, batchsize):

        return self.dst_iters.get_next_batch(batchsize)


if __name__ == "__main__":
    path = "ExpertData/"
    start_time = time.time()
    test_iters = Data_iters(path, [0], ['FrontRGB', 'FrontSemantic', 'speed', 'control', 'task_type'])
    end_time = time.time()
    print(end_time-start_time)

    b1, b2, b3, b4, b5= test_iters.get_next_branch(100)
    end_sample_time = time.time()
    print(end_sample_time - end_time)

    print("10: ",  b1.shape)
    print("11: ", b2, b2.shape)
    print("12: ", b3, b3.shape)
    print("13: ", b4, b4.shape)
    print("14: ",  b5, b5.shape)


