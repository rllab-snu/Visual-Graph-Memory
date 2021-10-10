import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ep-per-env', type=int, default=200, help='number of episodes per environments')
parser.add_argument('--num-procs', type=int, default=4, help='number of processes to run simultaneously')
parser.add_argument('--num-goals', type=int, default=5, help='number of goals per episodes')
parser.add_argument("--gpu", type=str, default="0", help="gpus",)
parser.add_argument('--split', type=str, default="val", choices=['train','val'], help='data split to use')
parser.add_argument('--data-dir', type=str, default="./IL_data", help='directory to save the collected data')
args = parser.parse_args()
import os
os.environ['GLOG_minloglevel'] = "2"
os.environ['MAGNUM_LOG'] = "quiet"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import glob
import numpy as np
import habitat
import habitat.sims
import habitat.sims.habitat_simulator
import joblib
from configs.default import get_config
from env_utils.task_search_env import MultiSearchEnv
from tqdm import tqdm

CONTENT_PATH = os.path.join(habitat.__path__[0],'../data/datasets/pointnav/gibson/v1/train/content/')
NUM_EPISODE_PER_SPACE = args.ep_per_env

def make_env_fn(config_env, rank):
    config_env.defrost()
    config_env.SEED = rank * 1121
    config_env.freeze()
    env = MultiSearchEnv(config=config_env)
    env.seed(rank * 1121)
    return env

def data_collect(space, config, DATA_DIR):
    num_of_envs = args.num_procs
    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip([config] * num_of_envs, range(num_of_envs))
            )
        ),
        auto_reset_done=False
    )
    num_episodes = int(NUM_EPISODE_PER_SPACE/num_of_envs)

    with tqdm(total=num_episodes) as pbar:
        for episode in range(num_episodes):
            observations = envs.reset()
            episodes = envs.current_episodes()
            space_dirnames = []
            episode_names = []
            for idx, ep in enumerate(episodes):
                space_name = episodes[0].scene_id.split('/')[-1][:-4]
                space_dirnames.append(DATA_DIR)
                episode_num = int(ep.episode_id) * num_of_envs + idx
                episode_name = '%s_%03d' % (space_name, episode_num)
                episode_names.append(episode_name)

            datas = [{'rgb': [], 'position': [], 'rotation': [], 'action': [], 'depth': [], 'target_idx': [], 'target_img': None, 'target_pose': None, 'distance': []} for _ in range(num_of_envs)]
            step = 0
            dones = envs.call(['get_episode_over']*num_of_envs)
            paused = [False] * num_of_envs
            env_ind_states = np.arange(num_of_envs)
            for i in range(num_of_envs):
                datas[i]['target_img'] = []
                datas[i]['target_pose'] = []
                for e in range(len(episodes[i].goals)):
                    datas[i]['target_img'].append(observations[i]['target_goal'][e])
                    datas[i]['target_pose'].append(episodes[i].goals[e].position)
            past_alive_indices = np.where(np.array(paused) == False)
            while (np.array(dones) == 0).any():
                best_actions = np.array(envs.call(['get_best_action']*num_of_envs))
                curr_goal_indices = envs.call(['get_curr_goal_index']*num_of_envs)
                alive_indices = np.where(np.array(paused) == False)
                past_obs = observations

                best_actions[np.where(best_actions == None)] = 0
                best_actions[np.where(envs.call(['get_episode_over']*num_of_envs)) == 1] = 0
                outputs = envs.step(best_actions)
                observations, rewards, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]
                for i, j in enumerate(past_alive_indices[0]):
                    # print(i, best_actions[0],best_actions[1],best_actions[2])
                    # if best_actions[i] == 0 : continue
                    # try:
                    #     assert episodes[j].start_position == past_obs[i]['episode_id'].start_position
                    #     assert (episodes[j].start_rotation == past_obs[i]['episode_id'].start_rotation).all()
                    #     assert episodes[j].goals[0].position == past_obs[i]['episode_id'].goals[0].position
                    # except:
                    #     print('where!!!!!!!!!!!!')
                    datas[j]['rgb'].append(past_obs[i]['panoramic_rgb'])
                    datas[j]['depth'].append(past_obs[i]['panoramic_depth'])
                    datas[j]['position'].append(past_obs[i]['position'])
                    datas[j]['rotation'].append(past_obs[i]['rotation'])
                    datas[j]['distance'].append(past_obs[i]['distance'])
                    if j in alive_indices[0]:
                        datas[j]['action'].append(best_actions[alive_indices[0].tolist().index(j)])
                        datas[j]['target_idx'].append(curr_goal_indices[alive_indices[0].tolist().index(j)])
                    try:
                        if j in alive_indices[0] and dones[alive_indices[0].tolist().index(j)] == 1:
                            ind = np.where(env_ind_states == j)
                            envs.pause_at(ind[0][0])
                            env_ind_states = np.delete(env_ind_states, ind)
                            paused[j] = True
                            continue
                    except:
                        print('h')

                step += 1
                past_alive_indices = alive_indices

            envs.resume_all()
            successes = envs.call(['get_success']*num_of_envs)
            for i in range(num_of_envs):
                success = successes[i]
                if success:
                    joblib.dump(datas[i], os.path.join(space_dirnames[i], episode_names[i] + '_env{}.dat.gz'.format(i)))

            pbar.update(1)
            pbar.set_description('Total %05d, %s %03d/%03d data collected' % (len(os.listdir(space_dirnames[0])),
                                                                              space_name,
                                                                              len(glob.glob(os.path.join(space_dirnames[0], space_name) + '*')),
                                                                              NUM_EPISODE_PER_SPACE))
    envs.close()


from habitat import make_dataset
from env_utils.make_env_utils import add_panoramic_camera
def main():
    split = args.split
    DATA_DIR = args.data_dir
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
    DATA_DIR = os.path.join(DATA_DIR, split)
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

    config = get_config()
    habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), '../')
    config.defrost()
    config.RL.SUCCESS_DISTANCE = 0.5
    config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR)
    config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH)
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 300
    config.TASK_CONFIG.ENVIRONMENT.NUM_GOALS = 5
    config.TASK_CONFIG = add_panoramic_camera(config.TASK_CONFIG)
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["GOAL_INDEX"] + config.TASK_CONFIG.TASK.MEASUREMENTS
    config.TASK_CONFIG.TASK.GOAL_INDEX = config.TASK_CONFIG.TASK.SPL.clone()
    config.TASK_CONFIG.TASK.GOAL_INDEX.TYPE = 'GoalIndex'
    config.DIFFICULTY = 'random'
    config.noisy_actuation = False
    config.freeze()

    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    print(scenes)
    for space_id, space in enumerate(scenes):
        try:
            print('=' * 50)
            print('SPACE[%03d/%03d] STARTED %s' % (space_id, len(scenes), space))
            config.defrost()
            config.TASK_CONFIG.DATASET.CONTENT_SCENES = [space]
            config.freeze()
            data_collect(space, config, DATA_DIR)
        except:
            raise
            print('{} failed may be the space is too large or unexpected error'.format(space))
            unexpected_skip.append(space)
            print('unexpected_skipped envs : ', unexpected_skip)

if __name__ == "__main__":
    main()
