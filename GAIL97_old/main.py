import os
import time
import numpy as np
import tensorflow as tf
from collections import OrderedDict

from learner import Learner
from env.carla97_env import Env
from dataset.dataset_iters import Data_iters

from nn.image_agent import ImageAgent

from tools.logger import Logger
from tools.replaybuffer import ReplayBufferDict
from tools.others import OU, print_dict

from config import branch_list, get_scene_config


def test(sess, args, env, scene, agent, branch):
    s_t = env.reset(scene)
    
    video = []
    for j in range(args.rl_step):
        image_feature, lidar_input, measure_input, real_image = s_t

        a_t = agent.actor.predict_action(image_feature, lidar_input, measure_input, branch) + np.zeros([2])
        s_t1, _, done = env.step(a_t)

        if j % args.frame_skip == 0:
            video.append(real_image)
            #print('frame ', j, ': ', a_t)

        s_t = s_t1
        if done:
            break
    
    return env.close('test'), video


def train(sess, args, exp_name):
    img_agent = ImageAgent(sess)
    agent = Learner(sess, args, branch_list)

    sess.run(tf.global_variables_initializer())
    img_agent.load_model()

    exp_dataset = Data_iters(args.exp_path, [0], ['FrontRGB', 'Lidar', 'Measurement', 'control'], episode_ratio = args.exp_ratio, image_agent=img_agent, branch_list=branch_list)
    agent.assgin_exp_dataset(exp_dataset)

    saver = tf.train.Saver(max_to_keep=15)
    sess.graph.finalize()

    fp = os.path.dirname(__file__)
    if args.load_model != '':
        load_path = os.path.join(fp, 'ckpt', args.load_model)
        saver.restore(sess, load_path)
        print('# continue training from : ', load_path)

    env = Env(img_agent, args.port)
    logger = Logger(os.path.join(fp, 'log', exp_name))

    # ------------------ BC ------------------
    if args.bc_iters > 0:
        print('# Train BC', args.bc_iters)
        for i in range(args.bc_iters):
            print('============== Episode %d / %d =============' % (i, args.bc_iters))
            bc_losses = agent.train_bc() # TODO: output loss
            print(bc_losses)

            logger.log_scalar_one_plot('bc_losses', bc_losses, i)
            logger.flush()

            if (i+1) % args.bc_ck == 0:
                for branch in [0,1]:
                    scene = get_scene_config(args.scene, args.pose, branch, args.weather)
                    res, video = test(sess, args, env, scene, agent, branch)
                    logger.log_video(video, 'branch=' + str(branch), i)
                logger.flush()
                
                save_path = os.path.join(fp, 'ckpt', exp_name + '_BC%d' % i)
                saver.save(sess, save_path)
                print('# Save model to ', save_path)
    
    
    # ------------------ RL ------------------
    if args.rl_iters > 0:
        print('# Train RL', args.rl_iters)
        buffer_dict = ReplayBufferDict(args.replay_buffer, branch_list)
        epsilon = 1.0

        for i in range(args.rl_iters):
            print('========== EPISODE %d / %d============' % (i, args.rl_iters))

            tr_res = OrderedDict()
            for branch in branch_list:
                scene = get_scene_config(args.scene, args.pose, branch, args.weather)
                s_t = env.reset(scene)
                
                print('# Collect trajectory, step = %d, branch = %d' % (args.rl_step, branch))
                for j in range(args.rl_step):
                    image_feature, lidar_input, measure_input, _ = s_t
                    a_t_pridect = agent.actor.predict_action(image_feature, lidar_input, measure_input, branch)
                    epsilon -= 1.0 / args.explore

                    noise_t = np.zeros([2]) 
                    noise_t[0] = max(epsilon,0) * OU(a_t_pridect[0], 0.3 * branch, 0.2, 0.15) #steer
                    noise_t[1] = max(epsilon,0) * OU(a_t_pridect[1], 0.2, 0.2, 0.15) #speed
                    
                    a_t = a_t_pridect + noise_t
                    s_t1, r_t, done = env.step(a_t)

                    if j % args.frame_skip == 1: #??? 0
                        buffer_dict.add(branch, (s_t, a_t, [r_t], s_t1, [done]))

                    s_t = s_t1
                    if done:
                        break
                
                res = env.close('train')
                for (k, v) in res.items():
                    if not k in tr_res:
                        tr_res[k] = OrderedDict()
                    tr_res[k][str(branch)] = v
            
            for (k, v) in tr_res.items():
                logger.log_scalar_one_plot(k, v, i)
                print(k, v)
            
            # ------------------ Log & Train & Save ------------------
            ddpg_losses = agent.train_ddpg(buffer_dict)
            logger.log_scalar_one_plot('ddpg_losses', ddpg_losses, i)
            logger.flush()
            print(ddpg_losses)

            discriminator_losses = agent.train_discriminator(buffer_dict)
            logger.log_scalar_one_plot('discriminator_losses', discriminator_losses, i)
            
            print("Buffer: ", buffer_dict.size())
            #print_dict("Performance", tr_res)

            if (i+1) % args.rl_ck == 0:
                for branch in branch_list:
                    scene = get_scene_config(args.scene, args.pose, branch, args.weather)
                    res, video = test(sess, args, env, scene, agent, branch)
                    logger.log_video(video, 'branch=' + str(branch), i)
                logger.flush()

                save_path = os.path.join(fp, 'ckpt', exp_name + '_RL%d' % i)
                saver.save(sess, save_path)
                print('# Save model to ', save_path)


def main():
    import argparse
    argparser = argparse.ArgumentParser(description="GAIL for CARLA-0.9.7")

    # ------------------ Experiment Configuration ------------------
    argparser.add_argument('--task', default= 'RL', type=str, help='task name')
    argparser.add_argument('--port', default= 2000, type=int, help='carla port')

    argparser.add_argument('--scene', default= 0, type=int, help='scene ID')
    argparser.add_argument('--pose', default= 0, type=int, help='pose ID')
    #argparser.add_argument('--branch', default= 0, type=int, help='left -1, straight 0, right 1')
    argparser.add_argument('--weather', default = 'ClearNoon', type=str, help='weather in carla simulator')

    # ------------------ Model Configuration ------------------
    argparser.add_argument('--load_model', default='', type=str, help='file name of the checkpoint to load')    
    argparser.add_argument('--exp_path', default='dataset/', type=str, help='file path of the expert data')
    argparser.add_argument('--exp_ratio', default=1.0, type=float, help='ratio of the expert dataset')
    
    argparser.add_argument('--hid_dim', default=128, type=int, help='the hidden dimension of a & c & d network')
    argparser.add_argument('--gpu_fraction', default=0.5, type=float, help='gpu_memory_fraction of tensorflow')
    argparser.add_argument('--gpu_id', default='0', type=str, help='gpu ID for training')

    # ------------------ BC Trainer Hyperparameters ------------------
    ''' For test
    argparser.add_argument('--bc_iters', default=1, type=int, help='the number of episodes for pretraining')
    argparser.add_argument('--bc_batch', default=0, type=int, help='batch size of pretraining')
    argparser.add_argument('--bc_ck', default=1, type=int, help='checkpoint per certain number of episodes')
    argparser.add_argument('--bc_update', default=0, type=int, help='update times in each iter')
    '''

    argparser.add_argument('--bc_iters', default=100, type=int, help='the number of episodes for pretraining')
    argparser.add_argument('--bc_batch', default=32, type=int, help='batch size of pretraining')
    argparser.add_argument('--bc_ck', default=201, type=int, help='checkpoint per certain number of episodes')
    argparser.add_argument('--bc_update', default=200, type=int, help='update times in each iter')

    argparser.add_argument('--lr_bc', default=0.0001, type=float, help='learning rate for behavior cloning')

    # ------------------ RL Configuration ------------------
    argparser.add_argument('--rl_iters', default=0, type=int, help='number of episodes for RL training')
    argparser.add_argument('--rl_step', default=500, type=int, help='max steps for each episode')
    argparser.add_argument('--rl_batch', default=32, type=int, help='batch size for training')
    argparser.add_argument('--rl_ck', default=20, type=int, help='checkpoint per certain number of episodes')
    
    argparser.add_argument('--replay_buffer', default=100000, type=int, help='replay buffer size')
    argparser.add_argument('--frame_skip', default=2, type=int, help='only store j mod frame_skip == 0 into replay buffer')
    argparser.add_argument('--explore', default=100000, type=int, help='decay rate of explore')
    
    # ------------------ GAIL Trainer Hyperparameters ------------------
    argparser.add_argument('--ddpg_update', default=20, type=int, help='number of actor-critic updates each training')
    argparser.add_argument('--discriminator_update', default=40, type=int, help='number of discriminator updates each training')
    
    argparser.add_argument('--lam', default=1, type=float, help='combine reward')
    argparser.add_argument('--gamma', default=0.99, type=float, help='parameters for advantage function')
    argparser.add_argument('--tau', default=0.001, type=float, help='the momentum for updating target network')
    argparser.add_argument('--entcoeff', default=0.001, type=float, help='the coefficient for the entropy loss')

    argparser.add_argument('--lr_a', default=0.00001, type=float, help='learning rate for actor network')
    argparser.add_argument('--lr_c', default=0.001, type=float, help='learning rate for critic network')
    argparser.add_argument('--lr_d', default=0.0001, type=float, help='learning rate for discriminator network')


    args = argparser.parse_args()
    #assert args.bc_iters == 0 or args.rl_iters == 0, "You should train BC either RL"

    exp_name = args.task
    exp_name += '_scene%d_pose%d_%s' % (args.scene, args.pose, args.weather)
    exp_name += time.strftime('_%m%d_%H%M')
    print('# exp_name:', exp_name)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    config=tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as  sess:
        train(sess, args, exp_name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')
