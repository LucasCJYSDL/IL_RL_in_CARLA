import os
import numpy as np
from tensorboardX import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.sum_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)
        self.history = {}
        self.orz = {}

        print('# log path', log_dir)

    def log_scalar(self, name, scalar, step):
        if not (name in self.history):
            self.history[name] = []
        self.history[name].append(scalar)

        average = np.mean(self.history[name])
        self.sum_writer.add_scalar('{}'.format(name + '_Average'), average, step)

    def log_scalar_one_plot(self, name, scalar_dict, step):
        """Will log all scalars in the same plot."""

        if not name in self.orz:
            self.orz[name] = {}
        
        for (k, v) in scalar_dict.items():
            if not k in self.orz[name]:
                self.orz[name][k] = []
            self.orz[name][k].append(v)
        
        new_dict = {}
        for (k, v) in scalar_dict.items():
            new_dict[k + '_Average'] = np.mean(self.orz[name][k])

        self.sum_writer.add_scalars(name, new_dict, step)

    def log_scalar_dict(self, dic, step):
        for (k, v) in dic.items():
            self.log_scalar(k, v, step)
    
    def log_video(self, video, name, step, fps=20):
        video = np.array([np.transpose(video, [0, 3, 1, 2])])
        assert len(video.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"

        self.sum_writer.add_video('{}'.format(name), video, step, fps=fps)

    def flush(self):
        self.sum_writer.flush()
    

    '''


    def log_image(self, image, name, step):
        assert(len(image.shape) == 3)  # [C, H, W]
        self.sum_writer.add_image('{}'.format(name), image, step)
    
    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self.sum_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self.sum_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self.log_dir, "scalar_data.json") if log_path is None else log_path
        self.sum_writer.export_scalars_to_json(log_path)
    '''