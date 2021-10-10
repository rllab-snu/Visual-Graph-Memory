#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import random

import numpy as np
from configs.default import get_config
from trainer.rl.ppo.ppo_trainer_memory import PPOTrainer_Memory
#from trainer.rl import ppo
import env_utils
import os
os.environ['GLOG_minloglevel'] = "2"
os.environ['MAGNUM_LOG'] = "quiet"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="path to config yaml containing info about experiment",
)
parser.add_argument(
    "--version",
    type=str,
    required=True,
    help="version of the training experiment",
)
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    help="gpus",
)
parser.add_argument(
    "--stop",
    action='store_true',
    default=False,
    help="include stop action or not",
)
parser.add_argument(
    "--no-noise",
    action='store_true',
    default=False,
    help="include noise or not",
)
parser.add_argument(
    "--diff",
    default='hard',
    choices=['easy', 'medium', 'hard'],
    help="episode difficulty",
)
parser.add_argument(
    "--seed",
    type=str,
    default="none"
)
parser.add_argument(
    "--render",
    action='store_true',
    default=False,
    help="This will save the episode videos, periodically",
)

arguments = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu

def main():
    run_exp(**vars(arguments))

def run_exp(config: str, opts=None, *args, **kwargs) -> None:
    config = get_config(config, arguments.version)
    config.defrost()
    config.noisy_actuation = not arguments.no_noise
    config.DIFFICULTY = arguments.diff
    config.render = arguments.render

    if arguments.stop:
        config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    else:
        config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    if arguments.seed != 'none':
        config.TASK_CONFIG.SEED = int(arguments.seed)
    config.freeze()
    random.seed(config.TASK_CONFIG.SEED) 
    np.random.seed(config.TASK_CONFIG.SEED)
        
    trainer = PPOTrainer_Memory(config)
    trainer.train()


if __name__ == "__main__":
    main()
