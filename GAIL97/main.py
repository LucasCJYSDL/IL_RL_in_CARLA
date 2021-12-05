import os
import argparse
import tensorflow as tf

from learner import Learner


def main():
    parser = argparse.ArgumentParser(description="GAIL for CARLA-0.9.7")

    # ------------------ Important ------------------
    parser.add_argument('--task', type=str, help='task name')
    parser.add_argument('--load_data', default='', type=str, help='expert data')

    parser.add_argument('--bc_model', default='', type=str, help='load BC model')
    parser.add_argument('--rl_model', default='', type=str, help='load RL model')
    parser.add_argument('--gail_model', default='', type=str, help='load GAIL model')

    parser.add_argument('--bc_iters', default=0, type=int, help='training episodes for BC')
    parser.add_argument('--rl_iters', default=0, type=int, help='training episodes for RL')
    parser.add_argument('--gail_iters', default=0, type=int, help='training episodes for gail')

    parser.add_argument('--replay', default=4000, type=int, help='replay buffer size')
    
    # ----------------- carla and tensorflow ---------------
    parser.add_argument('--port', default= 2000, type=int, choices=[-1, 2000, 3000, 4000], help='carla port')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu ID for training')
    parser.add_argument('--max_to_keep', default=50, type=int, help='for saver')

    # ------------------ BC ------------------
    parser.add_argument('--bc_ck', default=100, type=int, help='checkpoint for BC')
    parser.add_argument('--bc_update', default=50, type=int, help='update times in each episode')
    parser.add_argument('--bc_batch', default=120, type=int, help='batch size of pretraining')
    
    parser.add_argument('--bc_dropout', default=0.5, type=float, help='keep prob')
    parser.add_argument('--bc_lr', default=0.0001, type=float, help='learning rate for behavior cloning')

    # ------------------ RL ------------------
    parser.add_argument('--rl_ck', default=5, type=int, help='checkpoint for RL')
    parser.add_argument('--rl_update', default=20, type=int, help='update times in each iter')
    parser.add_argument('--rl_batch', default=120, type=int, help='batch size')
    parser.add_argument('--rl_scenes', default=[0], nargs='+', type=int, help='scene ID for interaction')

    parser.add_argument('--rl_dropout', default=1.0, type=float, help='keep prob')
    parser.add_argument('--rl_gamma', default=0.9, type=float, help='parameters for advantage function')
    parser.add_argument('--rl_tau', default=0.001, type=float, help='the momentum for updating target network')
    
    parser.add_argument('--rl_lr_a', default=0.000001, type=float, help='lr for actor network')
    parser.add_argument('--rl_lr_c', default=0.0001, type=float, help='lr for critic network')

    # ----------------- GAIL -------------------
    parser.add_argument('--gail_ck', default=5, type=int, help='chekcpoint for GAIL')
    parser.add_argument('--gail_update', default=20, type=int, help='update times for gail')
    parser.add_argument('--gail_batch', default=120, type=int, help='batch size for gail')
    parser.add_argument('--gail_G', default=3, type=int, help='train G more than D')

    parser.add_argument('--gail_entcoeff', default=0.001, type=float, help='the coefficient for the entropy loss')
    parser.add_argument('--gail_lr_d', default=0.0001, type=float, help='lr for discriminator network')

    # ------------------ Tester Hyperparameters ------------------
    parser.add_argument('--eval_steps', default=500, type=int, help='RL env steps')
    parser.add_argument('--eval_save', default=2, type=int, help='save per certain frames')
    

    # ------------------ Begin ----------------------------------
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as  sess:
        learner = Learner(sess, args)

        if args.bc_iters > 0:
            learner.bc_learn(args)
        
        if args.rl_iters > 0:
            learner.rl_learn(args)
        
        if args.gail_iters > 0:
            learner.gail_learn(args)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')
