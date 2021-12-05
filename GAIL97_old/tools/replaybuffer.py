import random

from collections import deque, OrderedDict


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque(maxlen=buffer_size)

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def add(self, experience):
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
    
    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
    
    @property
    def count(self):
        return self.num_experiences


class ReplayBufferDict:

    def __init__(self, buffer_size, branch_list):
        self.branch_list = branch_list
        self.buffers = OrderedDict()

        for branch in branch_list:
            self.buffers[branch] = ReplayBuffer(buffer_size)
    
    def add(self, branch, experience):
        self.buffers[branch].add(experience)
    
    def count(self, branch):
        return self.buffers[branch].count
    
    def getBatch(self, branch, batch_size):
        return self.buffers[branch].getBatch(batch_size)
    
    def size(self):
        size_list = OrderedDict()
        for branch in self.branch_list:
            size_list[branch] = self.buffers[branch].count
        return size_list
    
    
