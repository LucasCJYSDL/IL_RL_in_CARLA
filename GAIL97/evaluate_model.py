import os
import time
import argparse
import numpy as np
import tensorflow as tf

from collections import OrderedDict

from nn.preprocessor import Preprocessor
from nn.actor import Actor

from env.carla97_env import Env
from tools.others import render_image, create_csv, write_csv

from configer import Configer, branch_list


class Evaluator:

    def __init__(self, sess, args):
        self.sess = sess
        self.env = Env(args.port)

        self.iters = args.iters
        self.lateral = args.lateral
        self.longitude = args.longitude

        self.fp = os.path.dirname(__file__)
        if self.lateral == 'NN' or self.longitude == 'NN':
            assert args.nn_model is not '', "You should specify model name"

            self.preprocessor = Preprocessor()
            self.actor = Actor(sess, self.preprocessor, branch_list)

            self.restore(args.nn_model)
            self.sess.graph.finalize()

        self.eval_steps = args.eval_steps
        self.eval_render = args.eval_render

        self.create_metrics()


    def restore(self, load_model):
        params = []
        for var in tf.trainable_variables():
            if var.name.split('/')[0] in ('Preprocessor', 'Actor'):
                params.append(var)
        
        saver = tf.train.Saver(var_list=params)

        load_path = os.path.join(self.fp, 'ckpt', load_model)
        saver.restore(self.sess, load_path)
        print('# restore Actor from : ', load_path)
    

    def create_metrics(self):
        csv_dir = os.path.join(self.fp, 'evaluation')
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)

        fn = 'Later_%s_Longi_%s' % (self.lateral, self.longitude)
        self.csv_path = os.path.join(csv_dir, fn + time.strftime('_%m%d_%H%M')+'.csv')
        
        self.first_write = True


    def evaluate_episode(self, scene):
        s_t, info_t = self.env.reset(scene)

        for i in range(self.eval_steps):
            image, lidar, measure = s_t

            if self.lateral == 'NN' or self.longitude == 'NN':
                a_t = self.actor.predict_action([image / 255], [lidar], [measure], scene['branch'], dropout=1.0)[0] + np.zeros([2])
            else:
                a_t = np.zeros([2])
            
            s_t1, __, done, info_t = self.env.step(a_t, lateral=self.lateral, longitude=self.longitude)
            
            if self.eval_render:
                #render_image('big_semantic', info_t['big_semantic'])
                #render_image('small_semantic', info_t['small_semantic'])
                render_image('FrontRGB', info_t['FrontRGB'])

            s_t = s_t1
            if done:
                break

        return self.env.res, self.env.close(self.eval_steps)


    def evaluate_scene(self, scene_list):
        for scene_id in scene_list:
            configer = Configer(scene_id)

            for pose in range(configer.poses_num()):
                for branch in configer.branches(pose):
                    scene = configer.scene_config(pose, branch)

                    res_list = OrderedDict()
                    for episode in range(self.iters):
                        print('# Evaluate scene_id = %d, pose = %d, branch = %d, Episode = %d / %d' % (scene_id, pose, branch, episode, self.iters))
                        res, _ = self.evaluate_episode(scene)
                        print(res)

                        for (k, v) in res.items():
                            if not k in res_list:
                                res_list[k] = []
                            res_list[k].append(v)

                    self.update_metrics(scene_id, pose, branch, res_list)


    def update_metrics(self, scene, pose, branch, res_list):
        res_mean = OrderedDict([('scene', scene), ('pose', pose), ('branch', branch)])
        for (k, v) in res_list.items():
            res_mean[k] = np.mean(v)
        
        if self.first_write:
            create_csv(self.csv_path, list(res_mean.keys()))
            self.first_write = False
        
        write_csv(self.csv_path, list(res_mean.values()))



def main():
    parser = argparse.ArgumentParser(description="Evaluate Model")

    # --------------------------- Important----------------
    parser.add_argument('--scenes', nargs='+', type=int, help='scene ID for evaluating')
    parser.add_argument('--iters', default=1, type=int, help='evaluate episodes')
    
    parser.add_argument('--lateral', type=str, choices=['NN', 'PID', 'PID_NOISE', 'TTC']) 
    parser.add_argument('--longitude', type=str, choices=['NN', 'PID', 'TTC'])
    parser.add_argument('--nn_model', default='', type=str, help='If you use NN, this term must be speficied')

    # --------------------------- Optional ---------------------------
    parser.add_argument('--port', default= 2000, type=int, help='carla port')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu ID for training')

    parser.add_argument('--eval_steps', default=1000, type=int, help='maximum env steps')
    parser.add_argument('--eval_render', default=True, type=bool, help='render semantic image')
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print('# scenes:', args.scenes)

    with tf.Session(config=config) as  sess:
        evaluator = Evaluator(sess, args)
        evaluator.evaluate_scene(scene_list= args.scenes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')


