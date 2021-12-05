import os
import time
import argparse
import numpy as np

from tools.buffer import Buffers
from tools.others import render_image

from env.carla97_env import Env
from configer import Configer, branch_list


class DataCollector:

    def __init__(self, args):
        self.env = Env(args.port)
        self.buffers = Buffers(args.buffer, branch_list, args.load_data)
        self.save_data = args.save_data

        self.iters = args.iters
        self.lateral = args.lateral
        self.longitude = args.longitude

        self.eval_steps = args.eval_steps
        self.eval_save = args.eval_save


    def collect_episode(self, scene, debug=True):
        s_t, _ = self.env.reset(scene)
        for i in range(self.eval_steps):
            s_t1, r_t, done, info_t = self.env.step(np.zeros([2]), lateral=self.lateral, longitude=self.longitude)
            a_t = info_t['a_t']

            if i % self.eval_save == 0:
                self.buffers.add(scene['branch'], (s_t, a_t, r_t, s_t1, [done]))

            if debug:
                #print(a_t, r_t)
                render_image("small_semantic", info_t['small_semantic'])
                render_image("big_semantic", info_t['big_semantic'])

            s_t = s_t1
            if done:
                break
        
        print('res = ', self.env.close(self.eval_steps))
        print('buffer count = ', self.buffers.count(-1), self.buffers.count(0), self.buffers.count(1))


    def collect_scene(self, scene_list):
        for episode in range(self.iters):
            for scene_id in scene_list:
                configer = Configer(scene_id)

                for pose in range(configer.poses_num()):
                    for branch in configer.branches(pose):
                        scene = configer.scene_config(pose, branch)

                        print('\n# Collect Data: Episode = (%d / %d), scene_id = %d, pose = %d, branch = %d, ' % (episode, self.iters, scene_id, pose, branch))
                        self.collect_episode(scene)

            if (episode + 1) % 10 == 0:
                self.buffers.save_to_h5py(self.save_data)



def main():
    parser = argparse.ArgumentParser(description="Data Collector using PID")

    # --------------------------- Important----------------
    parser.add_argument('--scenes', nargs='+', type=int, help='scene ID for collecting data') 
    parser.add_argument('--iters', type=int, help='trajector each pose and branch')
    parser.add_argument('--load_data', default='', type = str, help='continue collecting')

    # --------------------------- Optional ---------------------------
    parser.add_argument('--lateral', default='PID_NOISE', type=str, choices=['PID', 'PID_NOISE', 'TTC']) # You Must Specify It!
    parser.add_argument('--longitude', default='PID', type=str, choices=['PID', 'TTC']) # You Must Specify It!
    
    parser.add_argument('--port', default= 2000, type=int, help='carla port')
    parser.add_argument('--buffer', default= 100000, type=int, help='replay buffer size')

    parser.add_argument('--eval_steps', default=1000, type=int, help='maximum env steps')
    parser.add_argument('--eval_save', default=4, type=int, help='save (s_t, a_t, ...) pairs per certain interval')
    args = parser.parse_args()


    print('# scenes:', args.scenes)

    fn = 'scene' + ''.join([str(s) for s in args.scenes])
    args.save_data = os.path.join('./dataset',  fn + time.strftime('_%m%d_%H%M') + '.h5')
    print('# save path:', args.save_data)

    data_collector = DataCollector(args)
    data_collector.collect_scene(scene_list = args.scenes)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')