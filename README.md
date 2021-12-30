# Visual-Graph-Memory
This is an official GitHub Repository for paper "Visual Graph Memory with Unsupervised Representation for Visual Navigation", which is accepted as a regular paper (poster) in ICCV 2021.

## Setup
### Requirements
The source code is developed and tested in the following setting. 
- Python 3.6
- pytorch 1.4~1.7
- habitat-sim 0.1.7 (commit version: ee75ba5312fff02aa60c04f0ad0b357452fc2edc)
- habitat 0.1.7 (commit version: 34a4042c03d6596f1d614faa4891868ddaf81c04)

Please refer to [habitat-sim](https://github.com/facebookresearch/habitat-sim.git) and [habitat-lab](https://github.com/facebookresearch/habitat-lab.git) for installation.

To set the environment, run:
```
pip install -r requirements.txt
```


### Habitat Data (Gibson, MP3D) Setup

Most of the scripts in this code build the environments assuming that the **gibson/mp3d dataset** is in **habitat-lab/data/** folder.

The recommended folder structure of habitat-api (or habitat-lab):
```
habitat-api (or habitat-lab)
  └── data
      └── datasets
      │   └── pointnav
      │       └── gibson
      │           └── v1
      │               └── train
      │               └── val
      └── scene_datasets
          └── gibson_habitat
              └── *.glb, *.navmeshs  
```

otherwise, you should edit the data path in [these](https://github.com/rllab-snu/Visual-Graph-Memory/blob/4103038781211ed880894650e7aa7245ea627027/env_utils/make_env_utils.py#L110-L114) [lines](https://github.com/rllab-snu/Visual-Graph-Memory/blob/4103038781211ed880894650e7aa7245ea627027/env_utils/custom_habitat_env.py#L85-L92).

## VGM Demonstration
To visualize the VGM generation, run:
```
python vgm_demo.py --gpu 0 --num-proc 2
```
This command will show the online VGM generation during *random exploration*.
The rendering window will show the generated VGM and the observations as follows:

![vgm_demo_1](docs/vgm_demo_1.gif) ![vgm_demo_1](docs/vgm_demo_2.gif)

Note that the top-down map and pose information are only used for visualization, not for the graph generation. 


## Imitation Learning
1. Data generation
    ```
    python collect_IL_data.py --ep-per-env 200 --num-procs 4 --split train --data-dir /path/to/save/data
    ```
    This will generate the data for imitation learning.
    You can find some examples of the collected data in *IL_data* folder, and look into them with  *show_IL_data.ipynb*.
2. Training
    ```
   python train_bc.py --config configs/vgm.yaml --stop --gpu 0
    ```
3. Evaluation


## Reinforcement Learning
The reinforcement learning code is highly [based on habitat-lab/habitat_baselines](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines).
To train the agent with reinforcement learning (PPO), run:
```
python train_rl.py --config configs/vgm.yaml --version EXPERIMENT_NAME --diff hard --render --stop --gpu 0
```

