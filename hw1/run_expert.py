#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

#        with open(os.path.join('experts', args.envname + '.pkl'), 'wb') as f:
#            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
        
        #write
        # from https://github.com/Observerspy/CS294/blob/master/hw1/run_expert.py
        # get path 
        PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
        # path for saving training data
        save_dir = os.path.join(PROJECT_ROOT, "data/")
        # name of the file
        out = os.path.join(save_dir, args.envname+'.train')
        # savez函数的第一个参数是文件名，其后的参数都是需要保存的数组
        # https://blog.csdn.net/xiewenbo/article/details/73832027
        np.savez(out, expert_data['observations'], expert_data['actions'])
        
        #
        print("finished!")
if __name__ == '__main__':
    main()
