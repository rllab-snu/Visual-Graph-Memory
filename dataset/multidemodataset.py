import torch.utils.data as data
import numpy as np
import joblib
import torch
import quaternion as q

class HabitatDemoMultiGoalDataset(data.Dataset):
    def __init__(self, cfg, data_list, include_stop = False):
        self.data_list = data_list
        self.img_size = (64, 256)
        self.action_dim = 4 if include_stop else 3
        self.max_demo_length = cfg.BC.max_demo_length

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def get_dist(self, demo_position):
        return np.linalg.norm(demo_position[-1] - demo_position[0], ord=2)

    def pull_image(self, index):
        demo_data = joblib.load(self.data_list[index])
        scene = self.data_list[index].split('/')[-1].split('_')[0]
        start_pose = [demo_data['position'][0], demo_data['rotation'][0]]
        rotation = q.as_euler_angles(np.array(q.from_float_array(demo_data['rotation'])))[:,1]
        target_indices = np.array(demo_data['target_idx'])
        aux_info = {'have_been': None, 'distance': None}

        orig_data_len = len(demo_data['position'])
        start_idx = np.random.randint(orig_data_len - 10) if orig_data_len > 10 else 0
        end_idx = - 1

        demo_rgb = np.array(demo_data['rgb'][start_idx:end_idx], dtype=np.float32)
        demo_length = np.minimum(len(demo_rgb), self.max_demo_length)

        demo_dep = np.array(demo_data['depth'][start_idx:end_idx], dtype=np.float32)

        demo_rgb_out = np.zeros([self.max_demo_length, *demo_rgb.shape[1:]])
        demo_rgb_out[:demo_length] = demo_rgb[:demo_length]
        demo_dep_out = np.zeros([self.max_demo_length, *demo_dep.shape[1:]])
        demo_dep_out[:demo_length] = demo_dep[:demo_length]

        demo_act = np.array(demo_data['action'][start_idx:start_idx+demo_length], dtype=np.int8)
        demo_act_out = np.ones([self.max_demo_length]) * (-100)

        demo_act_out[:demo_length] = demo_act -1 if self.action_dim == 3 else demo_act
        targets = np.zeros([self.max_demo_length])
        targets[:demo_length] = target_indices[start_idx:start_idx+demo_length]
        target_img= demo_data['target_img']
        target_img_out = np.zeros([5, *target_img[0].shape])
        target_img_out[:len(target_img)] = target_img
        positions = np.zeros([self.max_demo_length,3])
        positions[:demo_length] = demo_data['position'][start_idx:start_idx+demo_length]
        rotations = np.zeros([self.max_demo_length])
        positions[:demo_length] = demo_data['position'][start_idx:start_idx+demo_length]
        rotations[:demo_length] = rotation[start_idx:start_idx+demo_length]
        have_been = np.zeros([self.max_demo_length])
        for idx, pos_t in enumerate(positions[start_idx:end_idx]):
            if idx == 0:
                have_been[idx] = 0
            else:
                dists = np.linalg.norm(positions[start_idx:end_idx][:idx - 1] - pos_t,axis=1)
                if len(dists) > 10:
                    far = np.where(dists > 1.0)[0]
                    near = np.where(dists[:-10] < 1.0)[0]
                    if len(far) > 0 and len(near) > 0 and (near < far.max()).any():
                        have_been[idx] = 1
                    else:
                        have_been[idx] = 0
                else:
                    have_been[idx] = 0
        aux_info['distance'] = np.zeros([self.max_demo_length])
        distances = np.maximum(1-np.array(demo_data['distance'][start_idx:start_idx+demo_length])/2.,0.0)
        aux_info['distance'][:demo_length] = torch.from_numpy(distances).float()
        aux_info['have_been'] = torch.from_numpy(have_been).float()
        return_tensor = [torch.from_numpy(demo_rgb_out).float(), torch.from_numpy(demo_dep_out).float(),
                         torch.from_numpy(demo_act_out).float(), torch.from_numpy(positions),torch.from_numpy(rotations), targets,
                         torch.from_numpy(target_img_out).float(), scene, start_pose, aux_info]
        return return_tensor

