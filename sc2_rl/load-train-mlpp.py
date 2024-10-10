# $ source ~/Desktop/sc2env/bin/activate

# so this works, so far. 

from stable_baselines3 import PPO, A2C
import os
from sc2env import Sc2Env
import time
from wandb.integration.sb3 import WandbCallback
import wandb


LOAD_MODEL = "models/1669468885/7000.zip"
# Environment:
env = Sc2Env()

# load the model:
model = A2C.load(LOAD_MODEL, env=env)

model_name = f"{int(time.time())}"

models_dir = f"models/{model_name}/"
logdir = f"logs/{model_name}/"


conf_dict = {"Model": "load-v1",
             "Machine": "Main",
             "policy":"MlpPolicy",
             "model_save_name": model_name,
             "load_model": LOAD_MODEL
             }

run = wandb.init(
    project=f'SC2RLv1',
    entity="danielemos",
    config=conf_dict,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    save_code=True,  # save source code
)


# further train:
TIMESTEPS = 1000
iters = 0
while True:
	print("On iteration: ", iters)
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")