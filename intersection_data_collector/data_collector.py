import os
import time
import argparse
import numpy as np

from tools.buffer import Buffers
from tools.others import render_image
import h5py
from env.carla97_env import Env
from configer import Configer, branch_list


class DataCollector:

    def __init__(self, args):
        self.env = Env(args.port)
        self.term_list = ['sem_image', 'rgb_image', 'lidar', 'lidar_raw', 'relative_acc', 'min_dis', 'relative_angle', 'relative_dis', 'velocity', 'location', 'rotation', 'control']
        self.buffers = Buffers(self.term_list)
        self.save_data = args.save_data
        self.iters = args.iters
        self.lateral = args.lateral
        self.longitude = args.longitude
        self.eval_steps = args.eval_steps
        self.eval_save = args.eval_save


    def collect_scene(self, scene_list, debug=False):

        f = h5py.File(self.save_data, 'a')
        for scene_id in scene_list:
            configer = Configer(scene_id)
            sub_scene = f.create_group('scene_'+str(scene_id))
            for pose in range(configer.poses_num()):
                sub_pose = sub_scene.create_group('pose_'+str(pose))
                for branch in configer.branches(pose):
                    scene = configer.scene_config(pose, branch)
                    sub_branch = sub_pose.create_group('branch_'+str(branch))
                    for episode in range(self.iters):
                        print('\n# Collect Data: Episode = (%d / %d), scene_id = %d, pose = %d, branch = %d, ' % (episode, self.iters, scene_id, pose, branch))
                        sub_episode = sub_branch.create_group('episode_'+str(episode))
                        ###########episode############
                        s_t, _, set_up = self.env.reset(scene)
                        if episode==0:
                            sub_branch.create_dataset('start_point', data=set_up[0])
                            sub_branch.create_dataset('end_point', data=set_up[1])
                            sub_branch.create_dataset('reference_path', data=set_up[2])
                            sub_branch.create_dataset('lane_type', data=scene['lane_type'])

                        for i in range(self.eval_steps):
                            s_t1, r_t, done, info_t = self.env.step(np.zeros([2]), lateral=self.lateral, longitude=self.longitude)
                            a_t = info_t['a_t']
                            if i % self.eval_save == 0:
                                self.buffers.add((s_t, a_t))
                            if debug:
                                # print(a_t, r_t)
                                render_image("small_semantic", info_t['small_semantic'])
                                render_image("big_semantic", info_t['big_semantic'])
                                render_image("small_rgb", info_t['small_rgb'])
                                render_image("big_rgb", info_t['big_rgb'])
                            s_t = s_t1
                            if done:
                                break
                        metrics = self.env.close(self.eval_steps)
                        ###########episode############
                        print('res = ', metrics)
                        self.buffers.save_to_h5py(sub_episode)
                        sub_metrics = sub_episode.create_group('evaluation_metrics')
                        for k in metrics.keys():
                            sub_metrics.create_dataset(k, data=np.array([metrics[k]]))
        f.close()



def main():
    parser = argparse.ArgumentParser(description="Data Collector using PID")

    # --------------------------- You Have to Specify ----------------
    parser.add_argument('--scenes', nargs='+', type=int, help='scene ID for collecting data') 
    parser.add_argument('--iters', default= 2, type=int, help='trajector each pose and branch')
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

    fn = 'intersection'
    if not os.path.exists('./dataset'):
        os.mkdir('./dataset')
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