from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv

from utils.wrappers import ObsWrapper, CropResizeWrapper, MyRewardWrapper
from utils.duckie_wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper

import gym
import gym_duckietown

map_name = "Duckietown-small_loop-v0"

model_to_be_loaded = "a2cDuckietown-small_loop-v060000.0"

if __name__=="__main__":
    model = A2C.load("../models/"+model_to_be_loaded)
    env = gym.make(map_name)
    env = CropResizeWrapper(env)
    env = ObsWrapper(env)
    eval_env = MyRewardWrapper(env)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, reward, done, _ = eval_env.step(action)
        env.render(mode="human")
        if done and reward < 0:
            print("*** CRASHED ***")
            print(reward)
        elif done:
            print("SAFE")
            print(reward)