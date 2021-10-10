import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import imageio
import cv2
import copy
from trainer.il.bc_wrapper import BCWrapper

class BC_trainer(nn.Module):
    def __init__(self, cfg, agent):
        super().__init__()
        self.agent = agent
        self.env_wrapper = BCWrapper(cfg, cfg.BC.batch_size)
        self.feature_dim = cfg.features.visual_feature_dim
        self.torch_device = 'cpu' if cfg.TORCH_GPU_ID == '-1' else 'cuda'
        self.optim = optim.Adam(
            list(filter(lambda p: p.requires_grad,self.agent.parameters())),
            lr=cfg.BC.lr
        )
        self.localize_mode = 'pred'
        self.config = cfg
        self.env_setup_done = False

    def save(self,file_name=None, epoch=0, step=0):
        if file_name is not None:
            save_dict = {}
            save_dict['config'] = self.config
            save_dict['trained'] = [epoch, step]
            save_dict['state_dict'] = self.agent.state_dict()
            torch.save(save_dict, file_name)

    def forward(self, batch, train=True):
        demo_rgb, demo_depth, demo_act, positions, rotations, targets, target_img, scene, start_pose, aux_info = batch
        demo_rgb, demo_depth, demo_act = demo_rgb.to(self.torch_device), demo_depth.to(self.torch_device), demo_act.to(self.torch_device)
        target_img, positions, rotations = target_img.to(self.torch_device), positions.to(self.torch_device), rotations.to(self.torch_device)
        aux_info = {'have_been': aux_info['have_been'].to(self.torch_device),
                    'distance': aux_info['distance'].to(self.torch_device)}
        self.B = demo_act.shape[0]
        self.env_wrapper.B = demo_act.shape[0]
        self.env_wrapper.reset_all_memory(self.B)
        lengths = (demo_act > -10).sum(dim=1)

        T = lengths.max().item()
        hidden_states = torch.zeros(self.agent.net.num_recurrent_layers, self.B, self.agent.net._hidden_size).to(self.torch_device)
        actions = torch.zeros([self.B]).cuda()
        results = {'imgs': [], 'curr_node': [], 'node_list':[], 'actions': [], 'gt_actions': [], 'target': [], 'scene':scene[0], 'A': [], 'position': [],
                   'have_been': [], 'distance': [], 'pred_have_been': [], 'pred_distance': []}
        losses = []
        aux_losses1 = []
        aux_losses2 = []
        for t in range(T):
            masks = lengths > t
            if t == 0: masks[:] = False
            target_goal = target_img[torch.range(0,self.B-1).long(), targets[:,t].long()]
            pose_t = positions[:,t]
            obs_t = self.env_wrapper.step([demo_rgb[:,t], demo_depth[:,t], pose_t, target_goal, torch.ones(self.B).cuda()*t, (~masks).detach().cpu().numpy()])

            if t < lengths[0]:
                results['imgs'].append(demo_rgb[0,t].cpu().numpy())
                results['target'].append(target_goal[0].cpu().numpy())
                results['position'].append(positions[0,t].cpu().numpy())
                results['have_been'].append(aux_info['have_been'][0,t].cpu().numpy())
                results['distance'].append(aux_info['distance'][0,t].cpu().numpy())
                results['node_list'].append(copy.deepcopy(self.env_wrapper.graph.node_position_list[0]))
                results['curr_node'].append(self.env_wrapper.graph.last_localized_node_idx[0].cpu().numpy())

            gt_act = copy.deepcopy(demo_act[:, t])
            if -100 in actions:
                b = torch.where(actions==-100)
                actions[b] = 0
            (
                values,
                pred_act,
                actions_log_probs,
                hidden_states,
                actions_logits,
                preds,
                _
            ) = self.agent.act(
                obs_t,
                hidden_states,
                actions.view(self.B,1),
                masks.unsqueeze(1)
            )
            if not (gt_act == -100).all():
                loss = F.cross_entropy(actions_logits.view(-1,actions_logits.shape[1]),gt_act.long().view(-1))#, weight=action_weight)
                pred1, pred2 = preds
                valid_indices = gt_act.long() != -100
                aux_loss1 = F.binary_cross_entropy_with_logits(pred1[valid_indices].view(-1), aux_info['have_been'][valid_indices,t].float().reshape(-1))
                aux_loss2 = F.mse_loss(F.sigmoid(pred2)[valid_indices].view(-1), aux_info['distance'][valid_indices,t].float().reshape(-1))

                losses.append(loss)
                aux_losses1.append(aux_loss1)
                aux_losses2.append(aux_loss2)
                results['actions'].append(pred_act[0].detach().cpu().numpy())
                results['gt_actions'].append(int(gt_act[0].detach().cpu().numpy()))

            else:
                results['actions'].append(-1)
                results['gt_actions'].append(-1)
            results['pred_have_been'].append(F.sigmoid(pred1)[0].detach().cpu().numpy())
            results['pred_distance'].append(F.sigmoid(pred2)[0].detach().cpu().numpy())
            actions = demo_act[:,t].contiguous()

        action_loss = torch.stack(losses).mean()
        aux_loss1 = torch.stack(aux_losses1).mean()
        aux_loss2 = torch.stack(aux_losses2).mean()
        total_loss = action_loss + aux_loss1 + aux_loss2
        if train:
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        loss_dict = {}
        loss_dict['loss'] = action_loss.item()
        loss_dict['aux_loss1'] = aux_loss1.item()
        loss_dict['aux_loss2'] = aux_loss2.item()
        return results, loss_dict

    def visualize(self, result_dict, file_name, mode='train'):
        if mode == 'train':
            imgs = result_dict['imgs']
            target = result_dict['target']
            acts, gt_acts = result_dict['actions'], result_dict['gt_actions']
            if 'node_list' in result_dict:
                node_list, curr_node, position = result_dict['node_list'], result_dict['curr_node'], result_dict['position']
            if 'have_been' in result_dict:
                have_been, distance = result_dict['have_been'], result_dict['distance']
                pred_have_been, pred_distance = result_dict['pred_have_been'], result_dict['pred_distance']

            writer = imageio.get_writer(file_name + '.mp4', fps=15)
            for t in range(len(imgs)):
                view_im = imgs[t]
                target_im = target[t][:,:,:3] * 255
                view_im = np.concatenate([view_im, target_im],0).astype(np.uint8)
                view_im = cv2.resize(view_im,(256,128))
                cv2.putText(view_im, "t: %03d"%t + " act {} gt_act {}".format(acts[t], gt_acts[t]), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
                if 'node_list' in result_dict and len(result_dict['node_list']) > 0:
                    node_idx = np.linalg.norm(np.array(node_list[t]) - np.array(position[t]).reshape(1,-1), axis=1).argmin()
                    cv2.putText(view_im, "num_node : %d, curr_node: %d , gt_node:%d" % (len(node_list[t]), curr_node[t], node_idx), (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(view_im, "have_been: %.3f / %d     dist: %.3f/%.3f"%(pred_have_been[t], have_been[t], pred_distance[t], distance[t]),
                            (20, 40 + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                writer.append_data(view_im)
            writer.close()
        else:
            imgs = result_dict['imgs']
            writer = imageio.get_writer(file_name+'.mp4')
            w,h = imgs[-1].shape[0],imgs[-1].shape[1]
            for t in range(len(imgs)):
                view_im = cv2.resize(imgs[t],(h,w))
                writer.append_data(view_im)
            writer.close()
