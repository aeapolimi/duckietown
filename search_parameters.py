from pyvirtualdisplay import Display
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay

import optuna

import gym
import gym_duckietown

from wrappers import ObsWrapper

from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import A2C
from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.evaluation import evaluate_policy

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
 
map_name = "Duckietown-small_loop-v0" #@param ['Duckietown-straight_road-v0','Duckietown-4way-v0','Duckietown-udem1-v0','Duckietown-small_loop-v0','Duckietown-small_loop_cw-v0','Duckietown-zigzag_dists-v0','Duckietown-loop_obstacles-v0','Duckietown-loop_pedestrians-v0']
display = Display(visible=0, size=(1400, 900))
display.start()
env = gym.make(map_name, accept_start_angle_deg=4)
env = ObsWrapper(env)
 
def objective(trial):

    gamma = trial.suggest_float("learning_rate", 0.4, 0.99)

    model = A2C(
      CnnLstmPolicy,
      env,
      gamma=gamma,
      n_steps=5,
      learning_rate=0.0005, #def=0.0007
      lr_schedule='constant',
      verbose=0,
      tensorboard_log="./a2c_duckieloop/"
    )

    model.learn(total_timesteps=int(5e4))
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    print(gamma)
    print(mean_reward)

    return mean_reward


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)