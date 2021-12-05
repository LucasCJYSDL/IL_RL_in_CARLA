import os
import time
import numpy as np
import tensorflow as tf

from nn.preprocessor import Preprocessor
from nn.actor import Actor
from nn.critic import Critic
from nn.discriminator import Discriminator

from trainer.bc_trainer import BCTrainer
from trainer.rl_trainer import RLTrainer
from trainer.gail_trainer import GAILTrainer
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
        self.discrimintor = Discriminator(sess, self.preprocessor, branch_list)

        self.actor.set_training(args)
        self.critic.set_training(args)
        self.discrimintor.set_training(args)
        self.sess.run(tf.global_variables_initializer())

        # load model
        self.fp = os.path.dirname(__file__)

        if args.bc_model != '':
            self.restore_bc(args.bc_model)
        
        if args.rl_model != '':
            self.restore_rl(args.rl_model)
        
        if args.gail_model != '':
            self.restore_gail(args.gail_model)

        self.saver = tf.train.Saver(max_to_keep=args.max_to_keep)
        self.sess.graph.finalize()

        # task name
        self.task_name = args.task + time.strftime('_%m%d_%H%M')
        print('# task_name:', self.task_name)

        self.logger = Logger(os.path.join(self.fp, 'log', self.task_name))

    
    def bc_learn(self, args):
        self.exp_data = Buffers(None, branch_list, args.load_data)
        self.bc_trainer = BCTrainer(args, self.actor, self.exp_data, branch_list)
        self.tester = Tester(args, self.actor, None)

        for i in range(args.bc_iters):
            print('============== Learn BC - Episode (%d / %d) =============' % (i, args.bc_iters))

            bc_loss = self.bc_trainer.train()
            self.logger.log_scalar('bc_loss', bc_loss, i)
            self.logger.flush()

            if (i+1) % args.bc_ck == 0:
                self.save_model(i)
                
                for (scene_id, pose, branch) in ((0, 0, 1), (2, 1, -1), (0, 0, 0)):
                    res, video = self.tester.test(scene_id, pose, branch)
                    self.log(res, video, branch, i)
                    print(res)


    def rl_learn(self, args):
        self.replay = Buffers(args.replay, branch_list, args.load_data)
        self.rl_trainer = RLTrainer(args, self.actor, self.critic, self.replay, branch_list)
        self.tester = Tester(args, self.actor, self.replay)

        for i in range(args.rl_iters):
            print('============== Learn RL - Episode (%d / %d) =============' % (i, args.rl_iters))

            self.tester.collect_data()

            rwd, rl_loss = self.rl_trainer.train()
            self.logger.log_scalar('rwd', rwd, i)
            self.logger.log_scalar('rl_loss', rl_loss, i)
            self.logger.flush()

            if (i+1) % args.rl_ck == 0:
                for (scene_id, pose, branch) in ((0, 0, 1), (2, 1, -1), (0, 0, 0)):
                    res, video = self.tester.test(scene_id, pose, branch)
                    self.log(res, video, branch, i)
                    print(res)
    
                self.save_model(i)


    def gail_learn(self, args):
        self.exp_data = Buffers(None, branch_list, args.load_data)
        self.replay = Buffers(args.replay, branch_list, '')

        self.gail_trainer = GAILTrainer(args, self.actor, self.critic, self.discrimintor, self.exp_data, self.replay, branch_list)
        self.tester = Tester(args, self.actor, self.replay)

        for i in range(args.gail_iters):
            print('============== Learn GAIL - Episode (%d / %d) =============' % (i, args.gail_iters))

            self.tester.collect_data()

            rwd, rl_loss, d_loss, g_acc, e_acc = self.gail_trainer.train()

            self.logger.log_scalar('rwd', rwd, i)
            self.logger.log_scalar('rl_loss', rl_loss, i)
            self.logger.log_scalar('d_loss', d_loss, i)
            self.logger.log_scalar('g_acc', g_acc, i)
            self.logger.log_scalar('e_acc', e_acc, i)
            self.logger.flush()

            if (i+1) % args.gail_ck == 0:
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


    def restore_gail(self, gail_model):
        params = []
        for var in tf.trainable_variables():
            if var.name.split('/')[0] in ('Preprocessor', 'Actor', 'TargetActor', 'Critic', 'TargetCritic', 'Discriminator'):
                params.append(var)
        
        saver = tf.train.Saver(var_list=params)

        load_path = os.path.join(self.fp, 'ckpt', gail_model)
        saver.restore(self.sess, load_path)
        print('# restore GAIL model from : ', load_path)


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
