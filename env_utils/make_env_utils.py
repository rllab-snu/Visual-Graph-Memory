#!/usr/bin/env python3

import random
from typing import Type, Union

import habitat
from habitat import Config, Env, RLEnv, make_dataset
import habitat_sim


import os

import numpy as np
def make_env_fn(
    config: Config, env_class: Type[Union[Env, RLEnv]], rank: int, kwargs
) -> Union[Env, RLEnv]:

    config.defrost()
    config.record = config.render #and rank%4==0
    config.freeze()
    env = env_class(config=config)
    env.seed(rank)
    return env


def add_panoramic_camera(task_config, normalize_depth=True, has_target=True):
    num_of_camera = 360//task_config.SIMULATOR.RGB_SENSOR.HFOV
    assert isinstance(num_of_camera, int)
    angles = [2 * np.pi * idx/ num_of_camera for idx in range(num_of_camera-1,-1,-1)]
    half = num_of_camera//2
    angles = angles[half:] + angles[:half]
    use_semantic = 'PANORAMIC_SEMANTIC_SENSOR' in task_config.TASK.SENSORS
    use_depth = 'PANORAMIC_DEPTH_SENSOR' in task_config.TASK.SENSORS
    sensors_with_ids = []
    sensors = []
    for camera_idx in range(num_of_camera):
        curr_angle = angles[camera_idx]
        if curr_angle > 3.14:
            curr_angle -= 2 * np.pi
        new_camera_config = task_config.SIMULATOR.RGB_SENSOR.clone()
        new_camera_config.TYPE = "PanoramicPartRGBSensor"

        new_camera_config.ORIENTATION = [0, curr_angle, 0]
        new_camera_config.ANGLE = "{}".format(camera_idx)
        task_config.SIMULATOR.update({'RGB_SENSOR_{}'.format(camera_idx): new_camera_config})
        sensors.append('RGB_SENSOR_{}'.format(camera_idx))

        if use_depth:
            new_depth_camera_config = task_config.SIMULATOR.DEPTH_SENSOR.clone()
            new_depth_camera_config.TYPE = "PanoramicPartDepthSensor"
            new_depth_camera_config.ORIENTATION = [0, curr_angle, 0]
            new_depth_camera_config.ANGLE = "{}".format(camera_idx)
            new_depth_camera_config.NORMALIZE_DEPTH = normalize_depth
            task_config.SIMULATOR.update({'DEPTH_SENSOR_{}'.format(camera_idx): new_depth_camera_config})
            sensors.append('DEPTH_SENSOR_{}'.format(camera_idx))
        if use_semantic:
            new_semantic_camera_config = task_config.SIMULATOR.SEMANTIC_SENSOR.clone()
            new_semantic_camera_config.TYPE = "PanoramicPartSemanticSensor"
            new_semantic_camera_config.ORIENTATION = [0, curr_angle, 0]
            new_semantic_camera_config.ANGLE = "{}".format(camera_idx)
            task_config.SIMULATOR.update({'SEMANTIC_SENSOR_{}'.format(camera_idx): new_semantic_camera_config})
            sensors.append('SEMANTIC_SENSOR_{}'.format(camera_idx))

    task_config.SIMULATOR.AGENT_0.SENSORS = sensors
    sensor_dict = {'TYPE': 'PanoramicRGBSensor', 'WIDTH': task_config.SIMULATOR.RGB_SENSOR.HEIGHT * 4,
                 'HEIGHT': task_config.SIMULATOR.RGB_SENSOR_0.HEIGHT, 'NUM_CAMERA': num_of_camera,
                 'AGENT_ID': str(id)}
    task_config.TASK['PANORAMIC_SENSOR'] = habitat.Config()
    task_config.TASK['PANORAMIC_SENSOR'].update(sensor_dict)
    sensors_with_ids.append('PANORAMIC_SENSOR')
    if use_depth:
        task_config.TASK['PANORAMIC_DEPTH_SENSOR'] = task_config.TASK['PANORAMIC_SENSOR'].clone()
        task_config.TASK['PANORAMIC_DEPTH_SENSOR'].TYPE = 'PanoramicDepthSensor'
        task_config.TASK['PANORAMIC_DEPTH_SENSOR'].NORMALIZE_DEPTH = True
        task_config.TASK['PANORAMIC_DEPTH_SENSOR'].MIN_DEPTH = 0.0
        task_config.TASK['PANORAMIC_DEPTH_SENSOR'].MAX_DEPTH = 10.0
        sensors_with_ids.append('PANORAMIC_DEPTH_SENSOR')
    if use_semantic:
        task_config.TASK['PANORAMIC_SEMANTIC_SENSOR'] = task_config.TASK['PANORAMIC_SENSOR'].clone()
        task_config.TASK['PANORAMIC_SEMANTIC_SENSOR'].TYPE = 'PanoramicSemanticSensor'
        sensors_with_ids.append('PANORAMIC_SEMANTIC_SENSOR')
    if has_target:
        task_config.TASK.SENSORS = sensors_with_ids + ['CUSTOM_VISTARGET_SENSOR']
        task_config.TASK.CUSTOM_VISTARGET_SENSOR = habitat.Config()
        task_config.TASK.CUSTOM_VISTARGET_SENSOR.TYPE = 'CustomVisTargetSensor'
        task_config.TASK.CUSTOM_VISTARGET_SENSOR.NUM_CAMERA = num_of_camera
        task_config.TASK.CUSTOM_VISTARGET_SENSOR.WIDTH = task_config.SIMULATOR.RGB_SENSOR_0.HEIGHT * 4
        task_config.TASK.CUSTOM_VISTARGET_SENSOR.HEIGHT = task_config.SIMULATOR.RGB_SENSOR_0.HEIGHT
    else:
        task_config.TASK.SENSORS.remove('CUSTOM_VISTARGET_SENSOR')

    task_config.TASK.SUCCESS = habitat.Config()
    if "STOP" not in task_config.TASK.POSSIBLE_ACTIONS:
        task_config.TASK.SUCCESS.TYPE = "Success_woSTOP"
    else:
        task_config.TASK.SUCCESS.TYPE = "Success"
    task_config.TASK.SUCCESS.SUCCESS_DISTANCE = task_config.TASK.SUCCESS_DISTANCE
    task_config.TASK.DISTANCE_TO_GOAL.TYPE = 'Custom_DistanceToGoal'
    return task_config

from env_utils.env_wrapper import *
def construct_envs(config,env_class, mode='vectorenv', make_env_fn=make_env_fn, run_type='train', no_val=False, fix_on_cpu=False):

    num_processes, num_val_processes = config.NUM_PROCESSES, config.NUM_VAL_PROCESSES
    total_num_processes = num_processes + num_val_processes
    if no_val: num_val_processes = 0
    configs = []
    env_classes = [env_class for _ in range(total_num_processes)]

    habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), '../')
    config.defrost()
    config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR)
    config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH)
    config.freeze()

    eval_config = config.clone()
    eval_config.defrost()
    eval_config.TASK_CONFIG.DATASET.SPLIT = 'val'
    eval_config.freeze()

    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    training_scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        training_scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
        eval_scenes = dataset.get_scenes_to_load(eval_config.TASK_CONFIG.DATASET)

    if num_processes > 1:
        if len(training_scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(training_scenes) < num_processes:
            raise RuntimeError(
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )

    random.shuffle(training_scenes)

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(training_scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    eval_scene_splits = [[] for _ in range(num_val_processes)]
    if num_val_processes > 0 :
        for idx, scene in enumerate(eval_scenes):
            eval_scene_splits[idx % len(eval_scene_splits)].append(scene)
    else:
        eval_scenes = []

    scene_splits += eval_scene_splits
    print('Total Process %d = train %d + eval %d '%(total_num_processes, num_processes, num_val_processes))
    for i, s in enumerate(scene_splits):
        if i < num_processes:
            print('train_proc %d :'%i, s)
        else:
            print('eval_proc %d :' % i, s)

    assert sum(map(len, scene_splits)) == len(training_scenes+eval_scenes)

    for i in range(total_num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.DATASET.SPLIT = 'train' if i < num_processes else 'val'
        if len(training_scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]
        task_config = add_panoramic_camera(task_config,
                                           has_target='search' in proc_config.ENV_NAME.lower() or getattr(proc_config,'TASK_TYPE', True))

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )
        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = habitat_sim.cuda_enabled and not fix_on_cpu

        proc_config.freeze()
        configs.append(proc_config)

    if mode == 'vectorenv':
        envs = habitat.VectorEnv(
            make_env_fn=make_env_fn,
            env_fn_args=tuple(
                tuple(zip(configs, env_classes, range(total_num_processes), [{'run_type':run_type}]*total_num_processes))
            ),
        )

        envs = eval(configs[0].WRAPPER)(envs, configs[0])
        print('[make_env_utils] Using Vector Env Wrapper - ', configs[0].WRAPPER)
    else:
        envs = make_env_fn(configs[0] ,env_class, 0, { 'run_type': run_type})
    return envs
