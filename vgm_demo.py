import os
os.environ['GLOG_minloglevel'] = "2"
os.environ['MAGNUM_LOG'] = "quiet"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0", help='gpu ids to use, -1 indicates cpu.  e.g --gpu 0,1')
parser.add_argument('--graph-th', type=float, default=0.75)
parser.add_argument('--num-proc', type=int, default=2)
parser.add_argument('--split', type=str, default="val", choices=['train','val'], help='data split to use')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if __name__ == '__main__':
    import torch
    from configs.default import get_config
    from env_utils.task_search_env import SearchEnv
    from env_utils.make_env_utils import construct_envs, make_env_fn
    config = get_config()
    config.defrost()
    config.NUM_PROCESSES = args.num_proc
    config.TASK_CONFIG.DATASET.SPLIT = args.split
    config.TASK_CONFIG.TASK.TOP_DOWN_MAP.DRAW_SOURCE = False
    config.TASK_CONFIG.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = False
    config.TASK_CONFIG.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = False

    config.render = True
    config.render_map = True
    config.DIFFICULTY = 'hard'
    config.noisy_actuation = False

    config.WRAPPER = 'GraphWrapper'
    config.graph_th = args.graph_th
    if torch.cuda.device_count() <= 1:
        config.TORCH_GPU_ID = 0
        config.SIMULATOR_GPU_ID = 0

    config.freeze()

    env = construct_envs(config, SearchEnv, make_env_fn=make_env_fn)
    obs = env.reset()
    env.envs.call(["build_path_follower"]*env.B)
    env.envs.call(["set_random_goals"] * env.B)
    done = False
    imgs = []

    total_time_dict = {}
    iter = 0
    while True:
        acts = env.envs.call(['get_random_goal_action'] * env.B)

        actions = []
        for id, a in enumerate(acts):
            if a not in [None, 0]:
                actions.append(a)
            else:
                env.envs.call_at(id, 'set_random_goals')
                a = env.envs.call_at(id, 'get_random_goal_action')
                actions.append(a)

        obs, reward, done, info = env.step(actions)

        env.envs.render('human')
        iter += 1


