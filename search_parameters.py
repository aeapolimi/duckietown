from pyvirtualdisplay import Display
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay

import optuna
from optuna.integration.tensorboard import TensorBoardCallback

import gym
import gym_duckietown

from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import DDPG
from stable_baselines import A2C
from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.evaluation import evaluate_policy

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

from wrappers import ObsWrapper

map_name = "Duckietown-small_loop-v0" #@param ['Duckietown-straight_road-v0','Duckietown-4way-v0','Duckietown-udem1-v0','Duckietown-small_loop-v0','Duckietown-small_loop_cw-v0','Duckietown-zigzag_dists-v0','Duckietown-loop_obstacles-v0','Duckietown-loop_pedestrians-v0']
display = Display(visible=0, size=(1400, 900))
display.start()
env = gym.make(map_name, accept_start_angle_deg=4)
env = ObsWrapper(env)
tensorboard_callback = TensorBoardCallback("./optuna/", metric_name="optuna_mean_reward")

def model_DDPG(gamma: float, tensorboard="./optuna/"):
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
        param_noise=param_noise, #exploration noise
        action_noise=action_noise, #policy noise
        buffer_size=50000,
        tensorboard_log=tensorboard
        )

def model_A2C(gamma: float, tensorboard="./optuna/"):
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
 
def objective(trial: optuna.trial.Trial)  -> float:

  gamma = trial.suggest_float("gamma", 0.4, 0.99)
  
  model = model_DDPG(gamma)
  model.learn(total_timesteps=int(1e4))
  mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
  print(gamma)
  print(mean_reward)

  return mean_reward


if __name__ == "__main__":
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=20)