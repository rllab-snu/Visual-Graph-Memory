import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import imageio
from copy import deepcopy
import json
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num-episodes", type=int, default=1400)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--version-name", type=str, required=True)
parser.add_argument("--stop", action='store_true', default=False)
parser.add_argument("--diff", choices=['random', 'easy', 'medium', 'hard'], default='hard')
parser.add_argument("--split", choices=['val', 'train', 'min_val'], default='val')
parser.add_argument('--eval-ckpt', type=str, required=True)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--record', choices=['0','1','2','3'], default='0') # 0: no record 1: env.render 2: pose + action numerical traj 3: features
parser.add_argument('--th', type=str, default='0.75') # s_th
parser.add_argument('--record-dir', type=str, default='data/video_dir')

args = parser.parse_args()
args.record = int(args.record)
args.th = float(args.th)
import os
if args.gpu != 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['GLOG_minloglevel'] = "3"
os.environ['MAGNUM_LOG'] = "quiet"
os.environ['HABITAT_SIM_LOG'] = "quiet"
import numpy as np
import torch
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.gpu != 'cpu':
    torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.enable = True
torch.set_num_threads(5)
from env_utils.make_env_utils import add_panoramic_camera
import habitat
from habitat import make_dataset
from env_utils.task_search_env import SearchEnv
from configs.default import get_config, CN
import time
import cv2
import gzip
import quaternion as q
def eval_config(args):
    config = get_config(args.config)
    config.defrost()
    config.use_depth = config.TASK_CONFIG.use_depth = True
    config.DIFFICULTY = args.diff
    habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), '../')
    print(args.config)
    config.TASK_CONFIG = add_panoramic_camera(config.TASK_CONFIG, normalize_depth=True)
    config.TASK_CONFIG.DATASET.SPLIT = args.split if 'gibson' in config.TASK_CONFIG.DATASET.DATA_PATH else 'test'
    config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR)
    config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH)
    if 'COLLISIONS' not in config.TASK_CONFIG.TASK.MEASUREMENTS:
        config.TASK_CONFIG.TASK.MEASUREMENTS += ['COLLISIONS']
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    if config.TASK_CONFIG.DATASET.CONTENT_SCENES == ['*']:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    else:
        scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = scenes
    ep_per_env = int(np.ceil(args.num_episodes / len(scenes)))
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = ep_per_env
    if args.stop:
        config.ACTION_DIM = 4
        config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS= ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    else:
        config.ACTION_DIM = 3
        config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        config.TASK_CONFIG.TASK.SUCCESS.TYPE = "Success_woSTOP"
    config.freeze()
    return config

def load(ckpt):
    if ckpt != 'none':
        sd = torch.load(ckpt,map_location=torch.device('cpu'))
        state_dict = sd['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            if 'actor_critic' in key:
                new_state_dict[key[len('actor_critic.'):]] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        if 'config' in sd.keys():
            return (new_state_dict, sd['config'])
        return (new_state_dict,None)
    else:
        return (None, None)
from runner import *
#TODO: ADD runner in the config file e.g. config.runner = 'VGMRunner' or 'BaseRunner'
def evaluate(eval_config, ckpt):
    if args.record > 0:
        if not os.path.exists(os.path.join(args.record_dir, args.version_name)):
            os.mkdir(os.path.join(args.record_dir, args.version_name))
        VIDEO_DIR = os.path.join(args.record_dir, args.version_name + '_video_' + ckpt.split('/')[-1] + '_' +str(time.ctime()))
        if not os.path.exists(VIDEO_DIR): os.mkdir(VIDEO_DIR)
        if args.record > 1:
            OTHER_DIR = os.path.join(args.record_dir, args.version_name + '_other_' + ckpt.split('/')[-1] + '_' + str(time.ctime()))
            if not os.path.exists(OTHER_DIR): os.mkdir(OTHER_DIR)
    state_dict, ckpt_config = load(ckpt)

    if ckpt_config is not None:
        task_config = eval_config.TASK_CONFIG
        ckpt_config.defrost()
        task_config.defrost()
        ckpt_config.TASK_CONFIG = task_config
        ckpt_config.runner = eval_config.runner
        ckpt_config.AGENT_TASK = 'search'
        ckpt_config.DIFFICULTY = eval_config.DIFFICULTY
        ckpt_config.ACTION_DIM = eval_config.ACTION_DIM
        ckpt_config.memory = eval_config.memory
        ckpt_config.scene_data = eval_config.scene_data
        ckpt_config.WRAPPER = eval_config.WRAPPER
        ckpt_config.REWARD_METHOD = eval_config.REWARD_METHOD
        ckpt_config.ENV_NAME = eval_config.ENV_NAME
        for k, v in eval_config.items():
            if k not in ckpt_config:
                ckpt_config.update({k:v})
            if isinstance(v, CN):
                for kk, vv in v.items():
                    if kk not in ckpt_config[k]:
                        ckpt_config[k].update({kk: vv})
        ckpt_config.freeze()
        eval_config = ckpt_config
    print(eval_config.memory)
    eval_config.defrost()
    eval_config.th = args.th

    eval_config.record = False # record from this side , not in env
    eval_config.render_map = args.record > 0 or args.render or 'hand' in args.config
    eval_config.noisy_actuation = True
    eval_config.freeze()
    runner = eval(eval_config.runner)(eval_config, return_features=args.record>2)

    print('====================================')
    print('Version Name: ', args.version_name)
    print('Runner : ', eval_config.runner)
    print('Policy : ', eval_config.POLICY)
    print('Difficulty: ', eval_config.DIFFICULTY)
    print('Stop action: ', 'True' if eval_config.ACTION_DIM==4 else 'False')
    print('====================================')
    
    runner.eval()
    if torch.cuda.device_count() > 0:
        runner.cuda()
    #runner.load(state_dict)

    try:
        runner.load(state_dict)
    except:
        raise
        agent_dict = runner.agent.state_dict()
        new_sd = {k: v for k, v in state_dict.items() if k in agent_dict.keys() and (v.shape == agent_dict[k].shape)}
        agent_dict.update(new_sd)
        runner.load(agent_dict)

    env = eval(eval_config.ENV_NAME)(eval_config)

    env.habitat_env._sim.seed(args.seed)
    if runner.need_env_wrapper:
        env = runner.wrap_env(env,eval_config)

    val_scene_ep_list = glob.glob("image-goal-nav-dataset/val/*")
    global scene_ep_dict
    global self
    from habitat_sim.utils.common import quat_from_coeffs
    he = env.env.habitat_env
    scene_ep_dict = {}
    total_ep_num = 0
    for scene_file in val_scene_ep_list:
        with gzip.open(scene_file) as fp:
            episode_list = json.loads(fp.read())
        scene_name = scene_file.split('/')[-1][:-len('.json.gz')]
        scene_ep_dict[scene_name] = [ep for ep in episode_list if ep['info']['difficulty'] == args.diff]
        total_ep_num += len(scene_ep_dict[scene_name])
    print("Diff %s Total %d eps"%(args.diff, total_ep_num))
    total_episode_id = 0
    from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
    def next_episode(episode_id, scene_id):
        scene_name = scene_id.split('/')[-1][:-len('.glb')]
        if episode_id >= len(scene_ep_dict[scene_name]):
            return None, False
        else:
            episode_info = scene_ep_dict[scene_name][episode_id]
            new_episode = NavigationEpisode(**episode_info)
            new_episode.goals = [NavigationGoal(position=g['position']) for g in new_episode.goals]
            new_episode.start_rotation = q.as_float_array(quat_from_coeffs(new_episode.start_rotation))
            return new_episode, True
    env.env.habitat_env.get_next_episode_search = next_episode


    result = {}
    result['config'] = eval_config
    result['args'] = args
    result['version_name'] = args.version_name
    result['start_time'] = time.ctime()
    result['noisy_action'] = env.noise
    scene_dict = {}
    render_check = False
    with torch.no_grad():
        ep_list = []
        total_success, total_spl, total_node_dists, total_success_timesteps = [], [], [], []
        for episode_id in range(args.num_episodes):
            obs = env.reset()
            if render_check == False:
                if obs['panoramic_rgb'].sum() == 0 :
                    print('NO RENDERING!!!!!!!!!!!!!!!!!! YOU SHOULD CHECK YOUT DISPLAY SETTING')
                else:
                    render_check=True
            obs = runner.reset(obs)
            scene_name = env.current_episode.scene_id.split('/')[-1][:-4]
            if scene_name not in scene_dict.keys():
                scene_dict[scene_name] = {'success': [], 'spl': []}
            done = True
            reward = None
            info = None
            if args.record > 0:
                img = env.render('rgb')
                imgs = [img]
            step = 0
            while True:
                action = runner.step(obs, reward, done, info, env)
                obs, reward, done, info = env.step(action)
                step += 1
                if args.record > 0:
                    img = env.render('rgb')
                    imgs.append(img)
                if args.render:
                    env.render('human')
                if done: break
            spl = info['spl']
            if np.isnan(spl):
                spl = 0.0
            scene_dict[scene_name]['success'].append(info['success'])
            scene_dict[scene_name]['spl'].append(spl)
            total_success.append(info['success'])
            total_spl.append(spl)
            if info['success']:
                total_success_timesteps.append(step)
            #total_node_dists.append(np.array(node_dists).mean())
            ep_list.append({'house': scene_name,
                            'ep_id': env.current_episode.episode_id,
                            'start_pose': [env.current_episode.start_position, env.current_episode.start_rotation],
                            'target_pose': env.habitat_env.task.sensor_suite.sensors['target_goal'].goal_pose,
                            'total_step': step,
                            'collision': info['collisions']['count'] if isinstance(info['collisions'], dict) else info['collisions'],
                            'success': info['success'],
                            'spl': spl,
                            'distance_to_goal': info['distance_to_goal'],
                            'target_distance': env.habitat_env._sim.geodesic_distance(env.habitat_env.current_episode.goals[0].position,env.current_episode.start_position),})
            if args.record > 0:
                video_name = os.path.join(VIDEO_DIR,'%04d_%s_success=%.1f_spl=%.1f.mp4'%(episode_id, scene_name, info['success'], spl))
                with imageio.get_writer(video_name, fps=30) as writer:
                    im_shape = imgs[-1].shape
                    for im in imgs:
                        if (im.shape[0] != im_shape[0]) or (im.shape[1] != im_shape[1]):
                            im = cv2.resize(im, (im_shape[1], im_shape[0]))
                        writer.append_data(im.astype(np.uint8))
                    writer.close()
            print('[%04d/%04d] %s success %.4f, spl %.4f, total success %.4f, spl %.4f, success time step %.4f' % (episode_id,
                                                          args.num_episodes,
                                                          scene_name,
                                                          np.array(scene_dict[scene_name]['success']).mean(),
                                                          np.array(scene_dict[scene_name]['spl']).mean(),
                                                          np.array(total_success).mean(),
                                                          np.array(total_spl).mean(),
                                                          np.array(total_success_timesteps).mean()))
    result['detailed_info'] = ep_list
    result['each_house_result'] = {}
    success = []
    spl = []
    for scene_name in scene_dict.keys():
        mean_success = np.array(scene_dict[scene_name]['success']).mean()
        mean_spl = np.array(scene_dict[scene_name]['spl']).mean()
        result['each_house_result'][scene_name] = {'success': mean_success, 'spl': mean_spl}
        print('SCENE %s: success %.4f, spl %.4f'%(scene_name, mean_success,mean_spl))
        success.extend(scene_dict[scene_name]['success'])
        spl.extend(scene_dict[scene_name]['spl'])
    result['total_success'] = np.array(success).mean()
    result['total_spl'] = np.array(spl).mean()
    result['total_timesteps'] = np.array(total_success_timesteps)
    print('================================================')
    print('total success : %.2f'%(np.array(success).mean()))
    print('total spl : %.2f'%(np.array(spl).mean()))
    print('total timesteps : %.2f'%(np.array(total_success_timesteps).mean()))
    env.close()
    return result

if __name__=='__main__':
    import joblib
    import glob
    cfg = eval_config(args)
    if os.path.isdir(args.eval_ckpt):
        print('eval_ckpt ', args.eval_ckpt, ' is directory')
        ckpts = [os.path.join(args.eval_ckpt,x) for x in sorted(os.listdir(args.eval_ckpt))]
        ckpts.reverse()
    elif os.path.exists(args.eval_ckpt):
        ckpts = args.eval_ckpt.split(",")
    else:
        ckpts = [x for x in sorted(glob.glob(args.eval_ckpt+'*'))]
        ckpts.reverse()   
    print('evaluate total {} ckpts'.format(len(ckpts)))
    print(ckpts)
    for ckpt in ckpts:
        if 'ipynb' in ckpt or 'pt' not in ckpt: continue
        print('============================', ckpt.split('/')[-1], '==================')
        result = evaluate(cfg, ckpt)
        curr_hostname = os.uname()[1]
        eval_data_name = 'eval_result_{}.dat.gz'.format(curr_hostname)
        if os.path.exists(eval_data_name):
            data = joblib.load(eval_data_name)
            if args.version_name in data.keys():
                data[args.version_name].update({ckpt + '_{}'.format(time.time()): result})
            else:
                data.update({args.version_name: {ckpt + '_{}'.format(time.time()): result}})
        else:
            data = {args.version_name: {ckpt + '_{}'.format(time.time()): result}}
        joblib.dump(data, eval_data_name)
