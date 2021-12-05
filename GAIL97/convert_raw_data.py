import os
import time
import math
import h5py
import argparse
import numpy as np

from tools.buffer import Buffers
from configer import branch_list



def main():
    raw_data_list = []
    for fn in os.listdir('./raw_dataset'):
        if fn.endswith('.h5'):
           raw_data_list.append(os.path.join('./raw_dataset', fn))
    
    
    save_data = os.path.join('./dataset',  time.strftime('%m%d_%H%M') + '.h5')
    print('# save path:', save_data)
    
    buffers = Buffers(None, branch_list, '')

    for raw_data in raw_data_list:
        print(raw_data)

        with h5py.File(raw_data, 'r') as f:
            for scene in f:
                for pose in f[scene]:
                    for branch in f[scene][pose]:
                        g = f[scene][pose][branch]

                        for episode in g:
                            if not episode.startswith('episode'):
                                continue
                            
                            #for item in g[episode]:
                            #    print(item)
                            #print('')

                            sem_image = g[episode]['sem_image'][:]  # (88, 200, 3)
                            lidar = g[episode]['lidar'][:]          # (720)

                            velocity = g[episode]['velocity'][:]
                            speed = 3.6 * np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2 + velocity[:, 2]**2)
                            speed = (speed / 30.).reshape(-1, 1)

                            min_dis = g[episode]['min_dis'][:]
                            relative_angle = g[episode]['relative_angle'][:] / 1.57
                            relative_dis = g[episode]['relative_dis'][:]

                            length = len(sem_image)

                            lane_type = np.tile(g['lane_type'][:].reshape(1, 3), (length, 1))

                            #print('speed',speed.shape)
                            #print('min_dis',min_dis.shape)
                            #print('relative_angle',relative_angle.shape)
                            #print('relative_dis',relative_dis.shape)
                            #print('lane_type',lane_type.shape)

                            measure = np.concatenate([speed, min_dis, relative_angle, relative_dis, lane_type], axis = 1)

                            control = g[episode]['control'][:] # (None, 3)
                            action = np.zeros((length, 2))
                            action[:, 0] += control[:, 0]
                            action[:, 1] += control[:, 1] - control[:, 2]

                            gtw = {
                                'branch_-1' : -1,
                                'branch_0' : 0,
                                'branch_1' : 1,
                            }
                            branch_id = gtw[branch]

                            for t in range(length):
                                buffers.add(branch_id, ((sem_image[t], lidar[t], measure[t]), action[t], 0, (0, 0, 0), 0))

                            #print(sem_image.shape)
                            #print(lidar.shape)
                            #print(measure.shape)
                            #print(action.shape)


    #buffers.add(branch, (s_t, a_t, r_t, s_t1, [done]))
    buffers.save_to_h5py(save_data)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')