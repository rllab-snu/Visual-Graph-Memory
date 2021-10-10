import torch

TIME_DEBUG = False

# this wrapper comes after vectorenv
from env_utils.env_wrapper.graph import Graph
from env_utils.env_wrapper.env_graph_wrapper import GraphWrapper

class BCWrapper(GraphWrapper):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,exp_config, batch_size):
        self.is_vector_env = True
        self.num_envs = batch_size
        self.B = self.num_envs

        self.input_shape = (64, 256)
        self.feature_dim = 512
        self.scene_data = exp_config.scene_data
        self.torch = exp_config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU
        self.torch_device = 'cuda:' + str(exp_config.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'
        self.visual_encoder_type = getattr(exp_config, 'visual_encoder_type', 'unsupervised')
        self.visual_encoder = self.load_visual_encoder(self.visual_encoder_type, self.input_shape, self.feature_dim).to(self.torch_device)

        self.graph = Graph(exp_config, self.B, self.torch_device)
        self.th = 0.75
        self.num_agents = exp_config.NUM_AGENTS
        self.need_goal_embedding = 'wo_Fvis' in exp_config.POLICY
        self.localize_mode = 'predict'
        self.reset_all_memory()

    def step(self, batch):
        demo_rgb_t, demo_depth_t, positions_t, target_img, t, mask = batch
        obs_batch = {}
        obs_batch['step'] = t
        obs_batch['target_goal'] = target_img
        obs_batch['panoramic_rgb'] = demo_rgb_t
        obs_batch['panoramic_depth'] = demo_depth_t
        obs_batch['position'] = positions_t
        curr_vis_embedding = self.embed_obs(obs_batch)
        self.localize(curr_vis_embedding, obs_batch['position'].detach().cpu().numpy(), t, mask)
        global_memory_dict = self.get_global_memory()
        obs_batch = self.update_obs(obs_batch, global_memory_dict)
        obs_batch['curr_embedding'] = curr_vis_embedding
        return obs_batch
