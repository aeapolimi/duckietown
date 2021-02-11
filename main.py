from pyvirtualdisplay import Display
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay

import gym
import gym_duckietown

from wrappers import ObsWrapper

from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import A2C
from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.evaluation import evaluate_policy

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
 
def main():
    map_name = "Duckietown-small_loop-v0" #@param ['Duckietown-straight_road-v0','Duckietown-4way-v0','Duckietown-udem1-v0','Duckietown-small_loop-v0','Duckietown-small_loop_cw-v0','Duckietown-zigzag_dists-v0','Duckietown-loop_obstacles-v0','Duckietown-loop_pedestrians-v0']
    display = Display(visible=0, size=(1400, 900))
    display.start()
    env = gym.make(map_name, accept_start_angle_deg=4)
    env = ObsWrapper(env)
    
    model = A2C(
        CnnLstmPolicy,
        env,
        gamma=0.55, #Discount reward def=0.99
        n_steps=5,
        learning_rate=0.0001435, #def=0.0007
        lr_schedule='constant',
        verbose=0,
        tensorboard_log="./a2c_duckieloop/"
    )

    for time in range(10):
        model.learn(total_timesteps=int(2e4))
        model.save("models/a2c"+map_name+str(1e4*time))
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(f"#{time} Trained 10000 timesteps, mean_reward: {mean_reward}, std_reward: {std_reward}")
  
    ipythondisplay.clear_output(wait=True)

if __name__ == "__main__":
    main()