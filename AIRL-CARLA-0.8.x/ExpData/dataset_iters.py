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
        self.randomize = randomize
        self.num_pairs = len(labels_0)
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

        self.pointer = end

        return labels_0, labels_1, labels_2, labels_3, labels_4

class Data_iters(object):

    def __init__(self, path, file_num, is_shuffle=True):
        self.path = path
        self.file_num = file_num
        self.is_shuffle = is_shuffle
        self.get_dataset()
        self.split_into_branches()
        del self.exp_dataset_rgb, self.exp_dataset_target
        self.further_split()
        del self.rgb_branches, self.target_branches
        self.get_iters()
        del self.dataset_branches

    def get_dataset(self):
        cnt = 0
        self.exp_dataset_rgb, self.exp_dataset_target = [], []
        for filename in os.listdir(self.path):
            filedir = os.path.join(self.path, filename)
            # print("1: ", filedir)
            f = h5py.File(filedir, 'r')
            self.exp_dataset_rgb.extend(f["rgb"][:])
            self.exp_dataset_target.extend(f["targets"][:])
            cnt += 1
            if cnt >= self.file_num:
                break
        self.exp_dataset_rgb = np.array(self.exp_dataset_rgb)
        self.exp_dataset_target = np.array((self.exp_dataset_target))
        # print("2: ", self.exp_dataset_target, self.exp_dataset_target.shape)

    def split_into_branches(self):
        self.rgb_branches = []
        self.target_branches = []
        for ind in range(4):
            index = np.where(abs(self.exp_dataset_target[:, 24] - (ind+2))<1e-3)
            self.rgb_branches.append(self.exp_dataset_rgb[index])
            self.target_branches.append(self.exp_dataset_target[index])
        # print("4: ", self.target_branches[0][:, 24], self.target_branches[3][:, 24])

    def further_split(self):
        self.dataset_branches = []
        for ind in range(4):
            images = self.rgb_branches[ind]
            speeds = np.expand_dims(self.target_branches[ind][:, 10], axis=1)
            actions = self.target_branches[ind][:, 0:3]
            means = actions
            logstds = np.zeros_like(means) + np.log(1e-2)
            # print("5: ", images.shape)
            # print("6: ", speeds, speeds.shape)
            # print("7: ", actions, actions.shape)
            # print("8: ", means, means.shape)
            # print("9: ", logstds, logstds.shape)
            self.dataset_branches.append({"images": images, "speeds": speeds, "actions": actions, "means": means,
                                          "logstds": logstds})

    def get_iters(self):
        self.dataset_iters = []
        for ind in range(4):
            temp_dest = self.dataset_branches[ind]
            temp_iters = Dset(temp_dest["images"], temp_dest["speeds"], temp_dest["actions"], temp_dest["means"],
                              temp_dest["logstds"], self.is_shuffle)
            self.dataset_iters.append(temp_iters)

    def get_next_branch(self, branch_num, batchsize):

        return self.dataset_iters[branch_num].get_next_batch(batchsize)


if __name__ == "__main__":
    path = "../AgentHuman/SeqTrain"
    start_time = time.time()
    test_iters = Data_iters(path, 100)
    end_time = time.time()
    print(end_time-start_time)

    b1, b2, b3, b4, b5 = test_iters.get_next_branch(0,10)
    end_sample_time = time.time()
    print(end_sample_time - end_time)

    print("10: ",  b1.shape)
    print("11: ", b2, b2.shape)
    print("12: ", b3, b3.shape)
    print("13: ", b4, b4.shape)
    print("14: ", b5, b5.shape)

