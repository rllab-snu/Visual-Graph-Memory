# Visual-Graph-Memory
This repository will be updated soon.

This is an official GitHub Repository for paper "Visual Graph Memory with Unsupervised Representation for Visual Navigation", which is accepted as a regular paper (poster) in ICCV 2021.

## Setup
#### Requirements
The source code is developed and tested in the following setting. 
- Python 3.6
- pytorch 1.4~1.7
- habitat 0.1.7
- habitat-sim 0.1.7 

To set the environment, run:
```
pip install -r requirements.txt
```


#### Habitat Setup

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

otherwise, you should edit the data path in these lines.

## VGM Demonstration
To visualize the VGM generation, run:
```
python vgm_demo.py --gpu 0 --num-proc 2
```
This command will show the online VGM generation during the random exploration.
The rendering window will show the generated VGM and the observations as follows:

![vgm_demo_1](docs/vgm_demo_1.gif) ![vgm_demo_1](docs/vgm_demo_2.gif)

Note that the top-down map and pose information is are only used for visualization, not for the graph generation. 


## IL training code
1. Data generation
    ```
    python collect_IL_data.py --ep-per-env 200 --num-procs 4 --split train --data-dir /path/to/save/data
    ```
    This will generate the data for imitation learning.
    
    You can find some examples of the collected data in *IL_data* folder, and look into them with  *show_IL_data.ipynb* 





### TODO

- [x] VGM demo
- [ ] RL training code
- [ ] IL training code
- [ ] PCL training repo
- [x] collect IL data

