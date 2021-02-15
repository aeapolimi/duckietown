from pyvirtualdisplay import Display
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay

import gym
import gym_duckietown

from utils.wrappers import ObsWrapper
from utils.duckie_wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper

from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.evaluation import evaluate_policy

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
 
def model_DDPG(gamma: float, env, tensorboard="./ddpg_duckieloop/"):
    """
        Model DDPG

        :param gamma: (float) Reward discount
    """
    n_actions = env.action_space.shape[-1]
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))

    return DDPG(
            "CnnPolicy",
            env,
            gamma=gamma,
            verbose=0,
            tau=0.005,
            batch_size=32,
            param_noise=param_noise, #exploration noise
            action_noise=action_noise, #policy noise
            buffer_size=10000,
            tensorboard_log=tensorboard
            )

def model_A2C(gamma: float, env, tensorboard="./a2c_duckieloop/"):
    """
        Model A2C

        :param gamma: (float) Reward discount
    """
    return A2C(
        CnnLstmPolicy,
        env,
        gamma=gamma,
        n_steps=5,
        learning_rate=0.0005, #def=0.0007
        lr_schedule='constant',
        verbose=0,
        tensorboard_log=tensorboard
        )



def main():
    modello = "ddpg"
    map_name = "Duckietown-small_loop-v0" #@param ['Duckietown-straight_road-v0','Duckietown-4way-v0','Duckietown-udem1-v0','Duckietown-small_loop-v0','Duckietown-small_loop_cw-v0','Duckietown-zigzag_dists-v0','Duckietown-loop_obstacles-v0','Duckietown-loop_pedestrians-v0']
    display = Display(visible=0, size=(1400, 900))
    display.start()
    env = gym.make(map_name, accept_start_angle_deg=4)
    env = ObsWrapper(env)
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    # env = ImgWrapper(env)
    # env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    
    if modello == "ddpg":
        model = model_DDPG(gamma=0.99, env=env)
    if modello == "a2c":
        model = model_A2C(gamma=0.99, env=env)

    if True:
        model = DDPG.load("models/a2cDuckietown-small_loop-v090000.0")
        model.set_env(env)

    for time in range(40):
        model.learn(total_timesteps=int(2e4))
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        model.save("models/"+modello+map_name+str(1e4*time))
        print(f"#{time} Trained 10000 timesteps, mean_reward: {mean_reward}, std_reward: {std_reward}")
  
    ipythondisplay.clear_output(wait=True)

if __name__ == "__main__":
    main()