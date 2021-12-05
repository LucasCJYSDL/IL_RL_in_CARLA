"""
Tester is used to interact with RL env

test() for evaluate

collect_data() for RL training
"""

import numpy as np

from env.carla97_env import Env
from configer import Configer


class Tester:

    def __init__(self, args, actor, buffers):
        self.actor = actor
        self.buffers = buffers

        self.env = Env(args.port) if args.port != -1 else None
        self.scene_list = args.rl_scenes

        self.eval_steps = args.eval_steps
        self.eval_save = args.eval_save


    def test(self, scene_id, pose, branch):
        print('# Tester scene_id = %d, pose = %d, branch = %d' %(scene_id, pose, branch))

        configer = Configer(scene_id)
        scene = configer.scene_config(pose, branch)

        s_t, info_t = self.env.reset(scene)
        video = []

        for _ in range(self.eval_steps):
            image, lidar, measure = s_t

            a_t = self.actor.predict_action([image / 255], [lidar], [measure], branch, dropout=1.0)[0] + np.zeros([2])
            s_t1, __, done, info_t = self.env.step(a_t, lateral='NN', longitude='NN')
            
            video.append(info_t['small_semantic'])
            s_t = s_t1
            
            if done:
                break
        
        return self.env.close(self.eval_steps), video[::self.eval_save]


    def collect_data(self):
        for scene_id in self.scene_list:
            configer = Configer(scene_id)

            for pose in range(configer.poses_num()):
                for branch in configer.branches(pose):

                    #if branch != 1:
                    #    continue

                    scene = configer.scene_config(pose, branch)
                    print('\n# Intertact with environment: scene_id = %d, pose = %d, branch = %d, ' % (scene_id, pose, branch))

                    s_t, _ = self.env.reset(scene)
                    for i in range(self.eval_steps):
                        image, lidar, measure = s_t
                        a_t = self.actor.predict_action([image / 255], [lidar], [measure], scene['branch'], dropout=1.0)[0] + np.zeros([2])
                        a_t += np.random.randn(2) * 0.05 # exploration

                        s_t1, r_t, done, info_t = self.env.step(a_t, lateral='NN', longitude='NN')

                        '''
                        if i % 40 == 0:
                            rxd = 0.0
                            gt = measure[1]
                            rxd += (gt > 1.0) * (a_t[0] > - gt * 0.1) * -0.1 * gt
                            rxd += (gt < -1.0) * (a_t[0] < - gt * 0.1) * 0.1 * abs(gt)
                            
                            print('min_dis', gt)
                            print('a_t', a_t[0])
                            #print('rwd', abs(measure[1]) - abs(s_t1[2][1]))
                            print('')
                        '''

                        if i % self.eval_save == 0:
                            self.buffers.add(scene['branch'], (s_t, a_t, r_t, s_t1, [done]))

                        s_t = s_t1
                        if done:
                            break
                    
                    print('res = ', self.env.close(self.eval_steps))
                    print('buffer count = ', self.buffers.count(-1), self.buffers.count(0), self.buffers.count(1))

