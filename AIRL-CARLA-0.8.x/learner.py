from AIRL.policy import Policy
from AIRL.discriminator import Discriminator
from AIRL.value_net import ValueNet
import numpy as np
from baselines import logger
from baselines.common import fmt_row
from Utils.training_tools import compute_path_probs

class Learner(object):

    def __init__(self, sess, LR_D, LR_V, ENT, exp_dataset, image_agent):
        self.sess = sess
        self.policy = Policy(sess, entcoeff=ENT)
        self.discriminator = Discriminator(sess, LR_D)
        self.value_func = ValueNet(sess, LR_V)
        self.exp_dataset = exp_dataset
        self.image_agent = image_agent

    def train(self, buffer_dict, batch_size, branch, cg_damping, max_kl, cg_iters):

        batch = buffer_dict[branch].getBatch(batch_size)
        image = np.asarray([e["ob"][0:-2] for e in batch])
        speed = np.asarray([e["ob"][-2:-1] for e in batch])
        action = np.asarray([e["ac"] for e in batch])
        lprob = np.asarray([e["lprob"] for e in batch])
        tdlamret = np.asarray([e["tdlamret"] for e in batch])
        vpred = np.asarray([e["vpred"] for e in batch])
        atarg = np.asarray([e["atarg"] for e in batch])
        atarg = (atarg - atarg.mean()) / atarg.std() #???

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, self.discriminator.loss_name[branch]))
        labels = np.zeros((batch_size * 2, 1))
        labels[batch_size:] = 1.0

        image_expert, speed_expert, ac_expert, mean_expert, std_expert = self.exp_dataset.get_next_batch(branch, batch_size)
        image_expert = self.image_agent.pre_process(image_expert)

        lprob_expert = compute_path_probs(ac_expert, mean_expert, std_expert)
        lprob_batch = np.concatenate([lprob, lprob_expert], axis=0)
        d_losses = self.discriminator.train_branch(image, speed, action, image_expert, speed_expert, ac_expert, labels,
                                                   lprob_batch, branch)
        print("debug 1:", d_losses)
        # logger.log(fmt_row(13, np.mean(d_losses, axis=0)))
        logger.log(fmt_row(13, d_losses))

        # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")
        logger.log(fmt_row(13, self.policy.loss_name[branch]))

        g_losses = self.policy.train_branch(image, speed, action, atarg, branch, cg_damping, max_kl, cg_iters)
        print("debug 2:", g_losses)
        logger.log(fmt_row(13, g_losses))

        # ------------------ Update V ------------------
        logger.log("Optimizing Value...")
        logger.log(fmt_row(13, self.value_func.loss_name[branch]))

        v_losses = self.value_func.train_branch(image, speed, tdlamret, branch)

        print("debug 2:", v_losses)
        logger.log(fmt_row(13, v_losses))

        return d_losses, g_losses, v_losses


