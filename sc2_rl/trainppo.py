from stable_baselines3 import PPO
import os
from sc2env import Sc2Env
import time
from wandb.integration.sb3 import WandbCallback
import wandb
from dotenv import load_dotenv


model_name = f"{int(time.time())}"

models_dir = f"models/{model_name}/"
logdir = f"logs/{model_name}/"


conf_dict = {"Model": "v1-ppo",
             "Machine": "Main",
             "policy":"MlpPolicy",
             "model_save_name": model_name}

# load credentials for logging
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

run = wandb.init(
    project=f'SC2RL',
    entity="danielemos",
    config=conf_dict,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    save_code=True,  # optional
)


if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = Sc2Env()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000
iters = 0
while True:
	print("On iteration: ", iters)
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
