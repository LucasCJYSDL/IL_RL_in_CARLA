import os
import h5py
import random
import numpy as np

from collections import deque, OrderedDict


class Buffers:

    def __init__(self, size, branch_list, load_data):
        self.buffers = OrderedDict()
        self.branch_list = branch_list

        for branch in branch_list:
            if size is None:
                self.buffers[branch] = deque()      # no length limitation
            else:
                self.buffers[branch] = deque(maxlen=size)
        
        if load_data != '':
            load_data = os.path.join('./dataset', load_data)
            self.load_from_h5py(load_data)
    

    def add(self, branch, item):
        '''(s_t, a_t, r_t, s_t1, done)'''
        
        self.buffers[branch].append(item)


    def count(self, branch):
        return len(self.buffers[branch])
    

    def get_batch(self, branch, batch_size):
        """return a dict(keys : img_t, lid_t, ...)"""
        samples = random.sample(self.buffers[branch], batch_size)
        return self.encoder(samples)


    def encoder(self, buffer):
        keys = ['img_t', 'lid_t', 'mea_t', 'a_t', 'r_t', 'img_t1', 'lid_t1', 'mea_t1', 'done']
        res = {key : [] for key in keys}

        for item in buffer:
            s_t, a_t, r_t, s_t1, done = item

            res['img_t'].append(s_t[0])
            res['lid_t'].append(s_t[1])
            res['mea_t'].append(s_t[2])

            res['a_t'].append(a_t)
            res['r_t'].append(r_t)

            res['img_t1'].append(s_t1[0])
            res['lid_t1'].append(s_t1[1])
            res['mea_t1'].append(s_t1[2])

            res['done'].append(done)

        for key in keys:
            res[key] = np.array(res[key])

        return res


    def save_to_h5py(self, filename):
        with h5py.File(filename, 'w') as f:
            for branch in self.branch_list:
                g = f.create_group('branch%d' % branch)
                res = self.encoder(self.buffers[branch])

                for k, v in res.items():
                    g.create_dataset(k, data=v)
        
        print('# Save Data successfully!')
        for branch in self.branch_list:
            print('Branch= %d, Count= %d' % (branch, self.count(branch)))


    def decoder(self, buffer, dataset):
        keys = ['img_t', 'lid_t', 'mea_t', 'a_t', 'r_t', 'img_t1', 'lid_t1', 'mea_t1', 'done']
        res = {}
        for key in keys:
            res[key] = dataset[key][:]

        cnt = len(res['img_t'])
        assert all(cnt == len(res[key]) for key in keys)

        for i in range(cnt):
            s_t = (res['img_t'][i], res['lid_t'][i], res['mea_t'][i])
            a_t = res['a_t'][i]
            r_t = res['r_t'][i]

            s_t1 = (res['img_t1'][i], res['lid_t1'][i], res['mea_t1'][i])
            done = res['done'][i]

            buffer.append((s_t, a_t, r_t, s_t1, done))


    def load_from_h5py(self, filename):
        with h5py.File(filename, 'r') as f:
            for branch in self.branch_list:
                self.decoder(self.buffers[branch], f['branch%d' % branch])
        
        print('# load data from %s successfully!' % filename)
        for branch in self.branch_list:
            print('Branch= %d, Count= %d' % (branch, self.count(branch)))
