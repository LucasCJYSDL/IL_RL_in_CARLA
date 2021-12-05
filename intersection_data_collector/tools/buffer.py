import os
import h5py
import random
import numpy as np



class Buffers:

    def __init__(self, term_list):
        self.buffers = {}
        self.term_list = term_list
        for term in term_list:
            self.buffers[term] = []

    def add(self, data):
        '''(s_t, a_t, r_t, s_t1, done)'''
        s_t, a_t = data
        for i in range(len(s_t)):
            term = self.term_list[i]
            self.buffers[term].append(s_t[i])
        self.buffers[self.term_list[-1]].append(a_t)

    def save_to_h5py(self, group):

        for key in self.buffers.keys():
            group.create_dataset(key, data=np.array(self.buffers[key]))
            self.buffers[key] = []


