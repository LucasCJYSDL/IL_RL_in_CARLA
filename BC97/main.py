import os
import argparse
import tensorflow as tf

from learner import Learner


def main():
    parser = argparse.ArgumentParser(description="GAIL for CARLA-0.9.7")

    # ------------------ Experiment Configuration ------------------
    parser.add_argument('--task', type=str, help='task name') #You have to specify it

    parser.add_argument('--port', default= 2000, type=int, choices=[-1, 2000], help='carla port')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu ID for training')

    # ------------------ Replay Buffer Configuration ------------------
    parser.add_argument('--load_data', default='', type =str, help='expert data')
    parser.add_argument('--buffer', default=8000, type=int, help='replay buffer size')
    
    # ------------------ Learner Configuration ------------------
    parser.add_argument('--bc_model', default='', type=str, help='BC model to load')
    parser.add_argument('--rl_model', default='', type=str, help='RL model to load')

    parser.add_argument('--bc_iters', default=0, type=int, help='training episodes for BC')
    parser.add_argument('--rl_iters', default=0, type=int, help='training episodes for RL')
    parser.add_argument('--rl_pre_iters', default=0, type=int, help='Critic initialization, small than rl_iters.')
 
    parser.add_argument('--bc_ck', default=100, type=int, help='episodes of checkpoint when training BC')
    parser.add_argument('--rl_ck', default=10, type=int, help='episodes of checkpoint when training RL')

    parser.add_argument('--max_to_keep', default=10, type=int, help='for tensorflow saver')

    # ------------------ BC Trainer Hyperparameters ------------------
    parser.add_argument('--bc_update', default=50, type=int, help='update times in each episode')
    parser.add_argument('--bc_batch', default=120, type=int, help='batch size of pretraining')
    
    parser.add_argument('--bc_dropout', default=0.5, type=float, help='keep prob')
    parser.add_argument('--bc_lr', default=0.0001, type=float, help='learning rate for behavior cloning')

    # ------------------ RL Trainer Hyperparameters ------------------
    parser.add_argument('--rl_update', default=200, type=int, help='update times in each iter')
    parser.add_argument('--rl_batch', default=120, type=int, help='batch size of pretraining')
    parser.add_argument('--rl_scenes', default=[0], nargs='+', type=int, help='scene ID for collecting data')

    parser.add_argument('--rl_dropout', default=1.0, type=float, help='keep prob')
    parser.add_argument('--rl_gamma', default=0.9, type=float, help='parameters for advantage function')
    parser.add_argument('--rl_tau', default=0.001, type=float, help='the momentum for updating target network')
    
    parser.add_argument('--rl_lr_a', default=0.00001, type=float, help='learning rate for actor network')
    parser.add_argument('--rl_lr_c', default=0.001, type=float, help='learning rate for critic network')

    # ------------------ Tester Hyperparameters ------------------
    parser.add_argument('--eval_steps', default=500, type=int, help='RL env steps')
    parser.add_argument('--eval_save', default=2, type=int, help='save video per certain frames')
    
    
    args = parser.parse_args()
    assert args.bc_iters == 0 or args.rl_iters == 0, 'You can only train BC or RL'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as  sess:
        learner = Learner(sess, args)

        if args.bc_iters > 0:
            learner.bc_learn()
        
        if args.rl_iters > 0:
            learner.rl_learn()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')
