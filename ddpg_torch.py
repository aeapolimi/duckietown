import os
import numpy as np

# Duckietown Specific
from learning.reinforcement.pytorch.ddpg import DDPG
from learning.utils.env import launch_env
from learning.reinforcement.pytorch.utils import seed, evaluate_policy, ReplayBuffer
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper

import gym
import gym_duckietown

map_name = "Duckietown-small_loop-v0" #@param ['Duckietown-straight_road-v0','Duckietown-4way-v0','Duckietown-udem1-v0','Duckietown-small_loop-v0','Duckietown-small_loop_cw-v0','Duckietown-zigzag_dists-v0','Duckietown-loop_obstacles-v0','Duckietown-loop_pedestrians-v0']

def _train(seeds, eval_freq, max_timesteps, save_models, expl_noise,
           batch_size, discount, tau, policy_noise, noise_clip, policy_freq,
           env_timesteps, replay_buffer_max_size, model_dir):   
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    display = Display(visible=0, size=(1400, 900))
    display.start()
    env = gym.make(map_name, accept_start_angle_deg=4)
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")
    
    # Set seeds
    seed(seeds)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    replay_buffer = ReplayBuffer(replay_buffer_max_size)
    print("Initialized DDPG")
    
    # Evaluate untrained policy
    evaluations= [evaluate_policy(env, policy)]
   
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    reward = 0
    episode_timesteps = 0
    
    print("Starting training")
    while total_timesteps < max_timesteps:
        
        print("timestep: {} | reward: {}".format(total_timesteps, reward))
            
        if done:
            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau)

                # Evaluate episode
                if timesteps_since_eval >= eval_freq:
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate_policy(env, policy))
                    print("rewards at time {}: {}".format(total_timesteps, evaluations[-1]))

                    if save_models:
                        policy.save(filename='ddpg', directory=model_dir)
                    np.savez("./results/rewards.npz",evaluations)

            # Reset environment
            env_counter += 1
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.predict(np.array(obs))
            if expl_noise != 0:
                action = (action + np.random.normal(
                    0,
                    expl_noise,
                    size=env.action_space.shape[0])
                          ).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)

        if episode_timesteps >= env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == env_timesteps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add(obs, new_obs, action, reward, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
    
    print("Training done, about to save..")
    policy.save(filename='ddpg', directory=model_dir)
    print("Finished saving..should return now!")


if __name__=="__main__":
    seeds = 0
    start_timesteps=1e4
    eval_freq=5e3 # How often (time steps) we evaluate
    max_timesteps=1e6  # Max time steps to run environment for
    save_models="store_true"  # Whether or not models are saved
    expl_noise=0.1  # Std of Gaussian exploration noise
    batch_size=32  # Batch size for both actor and critic
    discount=0.99  # Discount factor
    tau=0.005  # Target network update rate
    policy_noise=0.2  # Noise added to target policy during critic update
    noise_clip=0.5 # Range to clip target policy noise
    policy_freq=2  # Frequency of delayed policy updates
    env_timesteps=500  # Frequency of delayed policy updates
    replay_buffer_max_size=10000  # Maximum number of steps to keep in the replay buffer
    model_dir='learning/reinforcement/pytorch/models/'

    _train(
        seeds=seeds,
        eval_freq=eval_freq,
        max_timesteps=max_timesteps,
        save_models=save_models,
        expl_noise=expl_noise,
        batch_size=batch_size,
        discount=discount,
        tau=tau,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_freq=policy_freq,
        env_timesteps=env_timesteps,
        replay_buffer_max_size=replay_buffer_max_size,
        model_dir=model_dir
        )