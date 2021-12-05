import os
from Env.carla_env import Env
from ImageParser.image_parser import ImageAgent
from Utils.OU import OU
from Utils.replay_buffer import ReplayBuffer
from ExpData.dataset_iters import Data_iters

import tensorflow as tf
import pickle
import random
import logging
from Utils.evaluate import Eval
import argparse
from Utils.training_tools import *
from learner import Learner


def train(args, sess,image_agent,task_name, exp_dataset, continue_train=False):
    # ------------------ Params ------------------
    BUFFER_SIZE = args.buffer_size
    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    LAM = args.lam
    # INIT_LRA = 0.000001
    # INIT_LRC = 0.0001
    LR_V = args.vf_stepsize
    LR_D = args.d_stepsize
    ENT = args.entcoeff
    EPISODE_MAX_STEP = args.episode_max_step
    TOTAL_EPISODE = args.total_episode
    EXPLORE = args.explore
    EVAL_BATCH = args.eval_batch
    SAVE_BATCH = args.save_batch
    SAMPLE_BATCH = args.sample_batch

    # ------------------ Build Model ------------------
    agent = Learner(sess, LR_D, LR_V, ENT, exp_dataset, image_agent)

    # ------------------ Initialize ------------------
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()
    ou = OU()

    # ------------------ Load Model ------------------
    if continue_train:
        load_path = './checkpoint' + '/' + task_name + '_' +str(args.load_model_ind)
        saver = tf.train.Saver()
        saver.restore(sess, load_path)

    # ------------------ Make Buffer ------------------
    buffer_follow = ReplayBuffer(BUFFER_SIZE)
    buffer_straight = ReplayBuffer(BUFFER_SIZE)
    buffer_left = ReplayBuffer(BUFFER_SIZE)
    buffer_right = ReplayBuffer(BUFFER_SIZE)
    buffer_dict = {0:buffer_follow,1:buffer_left,2:buffer_right,3:buffer_straight}

    # ------------------ Get Env and Metrics Class ------------------
    epsilon = 1.0
    CURRENT_STEP = 0
    UPDATE_STEP = 0

    env = Env("./log",image_agent, city="/Game/Maps/Town01", save_images=False, continue_train = continue_train)
    eval = Eval("./eval", env)
    logging.info("START")

    for i in range(TOTAL_EPISODE):
        if i%EVAL_BATCH == 0:
            success_time = 0
        try:
            ob = env.reset()
        except Exception:
            continue
        total_reward = 0
        episode_step = 0
        s_t = ob
        new = True
        a_t = env.random_sample()
        trainable = False
        obs, news, acs, means, logstds, vpreds, true_rews, prev_acs, lprobs, rews = [], [], [], [], [], [], [], [], [], []

        # ------------------ Start an episode ------------------
        for j in range(EPISODE_MAX_STEP):
            if s_t is None or len(s_t)<514:
                break
            epsilon-=1.0/ EXPLORE
            image_input = s_t[0:-2]
            speed_input = s_t[-2:-1]
            #GO_STRAIGHT = 5.0,TURN_RIGHT = 4.0,TURN_LEFT = 3.0,LANE_FOLLOW = 2.0
            direction = s_t[-1:] 
            branch_st = int(direction-2)
#            if branch_st == -2:  # REACH_GOAL=0
#                break
            prev_ac = a_t
            # a_t=np.zeros([1,3]) #steer throttle brake
            # noise_t = np.zeros([1,3])
            a_t_predict, mean, logstd = agent.policy.pridect_output(image_input,speed_input,branch_st)
            # noise_t[0][0] = max(epsilon,0)*ou.function(a_t_predict[0][0],0,0.6,0.3)
            # noise_t[0][1] = max(epsilon,0)*ou.function(a_t_predict[0][0],0.5,1,0.1)
            # noise_t[0][2] = max(epsilon,0)*ou.function(a_t_predict[0][0],-0.1,1,0.05)
            # a_t = a_t_predict+noise_t
            a_t = a_t_predict
            vpred = agent.value_func.pridect_value(image_input, speed_input, branch_st)
            lprob = compute_path_probs(a_t, mean, logstd)
            rew = agent.discriminator.predict_reward(image_input, speed_input, a_t, lprob, branch_st)
            exp_ob, exp_new, exp_ac, exp_mean, exp_logstd, exp_vpred, exp_pac, exp_lprob, exp_rew = s_t, new, a_t[0], mean[0], logstd[0], vpred[0][0], prev_ac[0], lprob[0], rew[0]
            try:
                ob,r_t,new, ts = env.step(a_t[0])
                s_t1 = ob
                exp_rt = [r_t]
                if s_t1 is None or len(s_t1)<514:
                    j = j-1#???
                    print("Exception!!!")
                    continue
                obs.append(exp_ob)
                news.append(exp_new)
                acs.append(exp_ac)
                means.append(exp_mean)
                logstds.append(exp_logstd)
                vpreds.append(exp_vpred)
                prev_acs.append(exp_pac)
                true_rews.append(exp_rt)
                lprobs.append(exp_lprob)
                rews.append(exp_rew)
                # ------------------ Sample a batch successfully! ------------------
                if (j+1)%SAMPLE_BATCH==0 or new or j==EPISODE_MAX_STEP-1:
                    seg = {"ob": np.array(obs), "new": np.array(news), "ac": np.array(acs), "mean": np.array(means),
                           "logstd": np.array(logstds),  "vpred": np.array(vpreds), "true_reward": np.array(true_rews),
                           "nextvpred": vpred[0][0] * (1 - new), "prevac": np.array(prev_acs), "lprob": np.array(lprobs), "rew": np.array(rews)}
                    obs, news, acs, means, logstds, vpreds, true_rews, prev_acs, lprobs, rews = [], [], [], [], [], [], [], [], [], []
                    # ------------------ Get Advantage Function ------------------
                    add_vtarg_and_adv(seg, GAMMA, LAM)
                    # ------------------ Add Experience to Buffer ------------------
                    update_buffer(buffer_dict, seg)
                    trainable = True
            except Exception:
                print("Should break???")
                continue

            # train Actor and  Critic
            branch_to_train = random.choice([0,1,2,3])
            if buffer_dict[branch_to_train].count()>BATCH_SIZE and trainable:
                d_losses, glosses, v_losses = agent.train(buffer_dict, BATCH_SIZE, branch_to_train, args.cg_damping, args.max_kl, args.cg_iters)
                eval.showing_loss_on_board(UPDATE_STEP, d_losses, glosses, v_losses, branch_st)
                UPDATE_STEP += 1
                trainable = False

            total_reward+=r_t
            s_t = s_t1
            CURRENT_STEP+=1
            episode_step+=1
            if new:
                break

        # ------------------ Evaluation && Log ------------------
        total_timestamp = (ts - env.initial_timestamp)/1000.0
        env.recording.write_summary_results(i, env.id_experiment, env.start_point, env.weather, env.success,
                              env.initial_distance, env.distance, total_timestamp)
        assert len(env.control_ls) == len(env.measure_ls), "Errors occur when collectiong data!!!"
        env.recording.write_measurements_results(env.id_experiment, i, env.weather, env.start_point, env.measure_ls, env.control_ls)

        if env.success:
            success_time += 1
            logging.info('+++++ Accept in %f seconds! +++++', total_timestamp)
        else:
            logging.info('----- Failed! -----')

        if (i+1)%EVAL_BATCH == 0:
            success_ratio = float(success_time) / float(EVAL_BATCH)
            eval.showing_sucess_on_board(i, success_ratio)

        eval.showing_metrics_on_board(i)

        print("buffer lenth:{},{},{},{},total reward:{},current_step:{},total_step:{}, update_step:{}".format(buffer_dict[0].count(),
                    buffer_dict[1].count(),
                    buffer_dict[2].count(),
                    buffer_dict[3].count(),
                    total_reward,episode_step,CURRENT_STEP, UPDATE_STEP))

        # ------------------ Save Model ------------------
        if np.mod(i, SAVE_BATCH)==0:
            fname = os.path.join('./checkpoint', task_name)
            fname = fname + "_" + str(i)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.train.Saver()
            saver.save(sess,fname)
            with open("./episode.txt","w") as log:
                log.write(("{},{}\n").format(i,epsilon))
            # with open("./buffer.pkl","wb") as buffer_log:
            #     pickle.dump(buffer_dict, buffer_log)

    logging.info('----- Finished! -----')
    env.recording.log_end()

if __name__=="__main__":

    argparser = argparse.ArgumentParser(description="AIRL for CARLA-0.8")
    argparser.add_argument('--buffer_size', default=100000, type=int, help='buffer size for the branches')
    argparser.add_argument('--batch_size', default=128, type=int, help='batch size for training')
    argparser.add_argument('--episode_max_step', default=5000, type=int, help='max steps for each episode')
    argparser.add_argument('--total_episode', default=20000, type=int, help='number of episodes for training')
    argparser.add_argument('--explore', default=100000, type=int, help='decay rate')
    argparser.add_argument('--eval_batch', default=20, type=int, help='batch size for computing success ratio')
    argparser.add_argument('--continue_experiment', default=False, help='If you want to continue the experiment with the given log name')
    argparser.add_argument('--vf_stepsize', default=1e-3, type=float, help='learning rate for training value net')
    argparser.add_argument('--d_stepsize', default=1e-5, type=float, help='learning rate for training discriminator')
    argparser.add_argument('--entcoeff', default=1e-2, type=float, help='entropy coefficiency of policy')
    argparser.add_argument('--max_kl', type=float, default=0.01)
    argparser.add_argument('--cg_iters', type=int, default=10)
    argparser.add_argument('--cg_damping', type=float, default=0.1)
    argparser.add_argument('--exp_ind', type=int, default=0, help='index of the experienment on going')
    argparser.add_argument('--load_model_ind', type=int, default=-1, help='index of the checkpoint to load')
    argparser.add_argument('--save_batch', type=int, default=200, help='save per certain number of episodes')
    argparser.add_argument('--sample_batch', type=int, default=128, help='sample certain number of episodes')
    argparser.add_argument('--gamma', type=float, default=0.995, help='parameters for advantage function')
    argparser.add_argument('--lam', type=float, default=0.97, help='parameters for advantage function')
    argparser.add_argument('--exp_path', type=str, default="../AgentHuman/SeqTrain", help='path for expert data')
    argparser.add_argument('--data_num', type=int, default=1000, help='number of expert data files, 200 pieces of data for each file')
    args = argparser.parse_args()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config=tf.ConfigProto(gpu_options=gpu_options)
    task_name = "AIRL_CARLA_" + str(args.exp_ind)

    with tf.Session(config=config) as  sess:
        img_agent = ImageAgent(sess)
        img_agent.load_model()
        exp_dataset = Data_iters(args.exp_path, args.data_num)
        print("Loading expert data, you may have to wait for about one minute ... ")
        train(args, sess, img_agent, task_name, exp_dataset, args.continue_experiment)

             

    
