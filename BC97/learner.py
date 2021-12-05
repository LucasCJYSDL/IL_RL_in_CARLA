import os
import time
import numpy as np
import tensorflow as tf

from nn.preprocessor import Preprocessor
from nn.actor import Actor
from nn.critic import Critic

from trainer.bc_trainer import BCTrainer
from trainer.rl_trainer import RLTrainer
from trainer.tester import Tester

from tools.logger import Logger
from tools.buffer import Buffers

from configer import branch_list



class Learner:

    def __init__(self, sess, args):
        self.sess = sess
        
        # set up NN
        self.preprocessor = Preprocessor()
        self.actor = Actor(sess, self.preprocessor, branch_list)
        self.critic = Critic(sess, self.preprocessor, branch_list)

        self.actor.set_training(args)
        self.critic.set_training(args)
        self.sess.run(tf.global_variables_initializer())

        # load model
        self.fp = os.path.dirname(__file__)

        if args.bc_model != '':
            self.restore_bc(args.bc_model)
        
        if args.rl_model != '':
            self.restore_rl(args.rl_model)

        self.saver = tf.train.Saver(max_to_keep=args.max_to_keep)
        self.sess.graph.finalize()

        # task name
        self.task_name = args.task + time.strftime('_%m%d_%H%M')
        print('# task_name:', self.task_name)

        # Helper
        self.buffers = Buffers(args.buffer, branch_list, args.load_data)
        self.tester = Tester(args, self.actor, self.buffers)
        self.logger = Logger(os.path.join(self.fp, 'log', self.task_name))

        # Trainer
        self.bc_trainer = BCTrainer(args, self.actor, self.buffers, branch_list)
        self.rl_trainer = RLTrainer(args, self.actor, self.critic, self.buffers, branch_list)

        # set up hyperparameters
        self.bc_iters = args.bc_iters
        self.bc_ck = args.bc_ck

        self.rl_iters = args.rl_iters
        self.rl_pre_iters = args.rl_pre_iters
        self.rl_ck = args.rl_ck

    
    def bc_learn(self):
        for i in range(self.bc_iters):
            print('============== Learn BC - Episode (%d / %d) =============' % (i, self.bc_iters))

            bc_loss = self.bc_trainer.train()
            self.logger.log_scalar('bc_loss', bc_loss, i)
            self.logger.flush()

            if (i+1) % self.bc_ck == 0:
                for (scene_id, pose, branch) in ((0, 0, 1), (2, 1, -1)):
                    res, video = self.tester.test(scene_id, pose, branch)
                    self.log(res, video, branch, i)
                    print(res)
    
                self.save_model(i)


    def rl_learn(self):
        for i in range(self.rl_iters):
            print('============== Learn RL - Episode (%d / %d) =============' % (i, self.rl_iters))

            if i >= self.rl_pre_iters:
                self.tester.collect_data()

            rl_loss = self.rl_trainer.train(train_actor = (i >= self.rl_pre_iters))
            self.logger.log_scalar('rl_loss', rl_loss, i)
            self.logger.flush()

            if (i+1) % self.rl_ck == 0 and i >= self.rl_pre_iters:
                for (scene_id, pose, branch) in ((0, 0, 1), (2, 1, -1), (0, 0, 0)):
                    res, video = self.tester.test(scene_id, pose, branch)
                    self.log(res, video, branch, i)
                    print(res)
    
                self.save_model(i)


    def restore_bc(self, bc_model):
        params = []
        for var in tf.trainable_variables():
            if var.name.split('/')[0] in ('Preprocessor', 'Actor'):
                params.append(var)
        
        saver = tf.train.Saver(var_list=params)

        load_path = os.path.join(self.fp, 'ckpt', bc_model)
        saver.restore(self.sess, load_path)
        print('# restore BC model from : ', load_path)

        self.actor.run_assign_target(1.0)
        self.critic.run_assign_target(1.0) # copy NN to target NN


    def restore_rl(self, rl_model):
        params = []
        for var in tf.trainable_variables():
            if var.name.split('/')[0] in ('Preprocessor', 'Actor', 'TargetActor', 'Critic', 'TargetCritic'):
                params.append(var)
        
        saver = tf.train.Saver(var_list=params)

        load_path = os.path.join(self.fp, 'ckpt', rl_model)
        saver.restore(self.sess, load_path)
        print('# restore RL model from : ', load_path)


    def save_model(self, i):
        par_folder = os.path.join(self.fp, 'ckpt')
        if not os.path.exists(par_folder):
            os.mkdir(par_folder)

        folder = os.path.join(par_folder, self.task_name)
        if not os.path.exists(folder):
            os.mkdir(folder)

        save_path = os.path.join(folder, self.task_name + '_%d' % i)
        self.saver.save(self.sess, save_path)
        print('# Save model to ', save_path)
    

    def log(self, res, video, branch, i):
        self.logger.log_video(video, 'branch%d' % branch, i)
        for k in res:
            self.logger.log_scalar('branch%d_%s' % (branch, k), res[k], i)
