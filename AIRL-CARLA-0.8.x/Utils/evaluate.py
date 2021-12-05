import tensorflow as tf
from .statistics import stats
import numpy as np

def get_value(ind, val):
    value = [None, None, None, None]
    value[ind] = val
    return value

class Eval(object):

    def __init__(self, eval_dir, env):

        self.env = env
        self.writer = tf.summary.FileWriter(eval_dir)

    def showing_metrics_on_board(self, iter):

        metrics = ["Accumulated_reward", "Distance_to_target", "Run_step"]
        ev_stats = stats(metrics)
        ev_stats.add_all_summary(self.writer, [np.sum(self.env.reward_ls), self.env.distance,
                                               self.env.current_step], iter)

    def showing_sucess_on_board(self, iter, success_ratio):

        metrics = ["Success_ratio"]
        ev_stats = stats(metrics)

        ev_stats.add_all_summary(self.writer, [success_ratio], iter)

    def showing_loss_on_board(self, iter, d_losses, g_losses, v_losses, branch):

        metrics = ["loss_follow", "loss_left", "loss_right", "loss_straight"]
        d_stats = stats(["d_" + ln for ln in metrics])
        g_stats = stats(["g_" + ln for ln in metrics])
        v_stats = stats(["v_" + ln for ln in metrics])
        inds = [branch] * 3
        vals = [d_losses, g_losses, v_losses]
        res = list(map(get_value, inds,  vals))
        d_stats.add_all_summary(self.writer, res[0], iter)
        g_stats.add_all_summary(self.writer, res[1], iter)
        v_stats.add_all_summary(self.writer, res[2], iter)

    def plot_trajetory(self):
        pass

    #TODO

