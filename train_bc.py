import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from configs.default import get_config
from model.policy import *
from trainer.il.bc_trainer import BC_trainer
from gym.spaces.dict import Dict as SpaceDict
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import os
import argparse

from dataset.multidemodataset import HabitatDemoMultiGoalDataset
from torch.utils.data import DataLoader
import torch
torch.backends.cudnn.enable = True
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to config yaml containing info about experiment")
parser.add_argument("--gpu", type=str, default="0", help="gpus",)
parser.add_argument("--version", type=str, default="test", help="name to save")
parser.add_argument("--stop", action='store_true', default=False, help="include stop action or not",)
parser.add_argument('--data-dir', default='./IL_data', type=str)
parser.add_argument('--resume', default='none', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cpu' if args.gpu == '-1' else 'cuda'

def train():

    observation_space = SpaceDict({
        'panoramic_rgb': Box(low=0, high=256, shape=(64, 256, 3), dtype=np.float32),
        'panoramic_depth': Box(low=0, high=256, shape=(64, 256, 1), dtype=np.float32),
        'target_goal': Box(low=0, high=256, shape=(64, 256, 3), dtype=np.float32),
        'step': Box(low=0, high=500, shape=(1,), dtype=np.float32),
        'prev_act': Box(low=0, high=3, shape=(1,), dtype=np.int32),
        'gt_action': Box(low=0, high=3, shape=(1,), dtype=np.int32)
    })

    config = get_config(args.config)
    s = time.time()

    action_space = Discrete(4) if args.stop else Discrete(3)
    stop_info = 'INCLUDING' if args.stop else 'EXCLUDING'
    print('POLICY : {}'.format(config.POLICY))
    print('TRAINING INFO : {} STOP ACTION'.format(stop_info))

    config.defrost()
    config.NUM_PROCESSES = config.BC.batch_size
    config.TORCH_GPU_ID = args.gpu
    config.freeze()

    policy = eval(config.POLICY)(
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=config.features.hidden_size,
        rnn_type=config.features.rnn_type,
        num_recurrent_layers=config.features.num_recurrent_layers,
        backbone=config.features.backbone,
        goal_sensor_uuid=config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
        normalize_visual_inputs=True,
        cfg=config
    )
    trainer = eval(config.IL_TRAINER_NAME)(config, policy)


    DATA_DIR = args.data_dir
    train_data_list = [os.path.join(DATA_DIR, 'train', x) for x in sorted(os.listdir(os.path.join(DATA_DIR, 'train')))]
    valid_data_list = [os.path.join(DATA_DIR, 'val', x) for x in sorted(os.listdir(os.path.join(DATA_DIR, 'val')))]

    params = {'batch_size': config.BC.batch_size,
              'shuffle': True,
              'num_workers': config.BC.num_workers,
              'pin_memory': True}

    train_dataset = HabitatDemoMultiGoalDataset(config, train_data_list, args.stop)
    train_dataloader = DataLoader(train_dataset, **params)
    train_iter = iter(train_dataloader)

    valid_dataset = HabitatDemoMultiGoalDataset(config, valid_data_list, args.stop)
    valid_params = params

    valid_dataloader = DataLoader(valid_dataset, **valid_params)
    valid_iter = iter(valid_dataloader)

    version_name = config.saving.name if args.version == 'none' else args.version
    version_name = version_name
    version_name += '_start_time:{}'.format(time.ctime())

    IMAGE_DIR = os.path.join('data', 'images', version_name)
    SAVE_DIR = os.path.join('data', 'new_checkpoints', version_name)
    LOG_DIR = os.path.join('data', 'logs', version_name)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    start_step = 0
    start_epoch = 0
    if args.resume != 'none':
        sd = torch.load(args.resume)
        start_epoch, start_step = sd['trained']
        trainer.agent.load_state_dict(sd['state_dict'])
        print('load {}, start_ep {}, strat_step {}'.format(args.resume, start_epoch, start_step))


    print_every = config.saving.log_interval
    save_every = config.saving.save_interval
    eval_every = config.saving.eval_interval
    writer = SummaryWriter(log_dir=LOG_DIR)

    start = time.time()
    temp = start
    step = start_step
    step_values = [10000, 50000, 100000]
    step_index = 0
    lr = config.BC.lr

    def adjust_learning_rate(optimizer, step_index, lr_decay):
        lr = config.BC.lr * (lr_decay ** step_index)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    trainer.to(device)
    trainer.train()
    for epoch in range(start_epoch, config.BC.max_epoch):
        train_dataloader = DataLoader(train_dataset, **params)
        train_iter = iter(train_dataloader)
        loss_summary_dict = {}
        s = time.time()
        for batch in train_iter:
            results, loss_dict = trainer(batch)
            for k,v in loss_dict.items():
                if k not in loss_summary_dict.keys():
                    loss_summary_dict[k] = []
                loss_summary_dict[k].append(v)

            if step in step_values:
                step_index += 1
                lr = adjust_learning_rate(trainer.optim, step_index, config.training.lr_decay)

            if step % print_every == 0:
                loss_str = ''
                writer_dict = {}
                for k,v in loss_summary_dict.items():
                    value = np.array(v).mean()
                    loss_str += '%s: %.3f '%(k,value)
                    writer_dict[k] = value
                print("time = %.2fm, epo %d, step %d, lr: %.5f, %ds per %d iters || loss : " % ((time.time() - start) // 60, epoch + 1,
                                                                                                step + 1, lr, time.time() - temp, print_every), loss_str)
                loss_summary_dict = {}
                temp = time.time()
                writer.add_scalars('loss', writer_dict, step)
                trainer.visualize(results, os.path.join(IMAGE_DIR, 'train_{}_{}'.format(results['scene'],step)))

            if step % save_every == 0 :
                trainer.save(file_name=os.path.join(SAVE_DIR, 'epoch%04diter%05d.pt' % (epoch, step)),epoch=epoch, step=step)

            if step % eval_every == 0 and step > 0:
                trainer.eval()
                eval_start = time.time()
                with torch.no_grad():
                    val_loss_summary_dict = {}
                    for j in range(100):
                        try:
                            batch = next(valid_iter)
                        except:
                            valid_dataloader = DataLoader(valid_dataset, **valid_params)
                            valid_iter = iter(valid_dataloader)
                            batch = next(valid_iter)
                        results, loss_dict = trainer(batch, train=False)
                        if j % 100 == 0:
                            trainer.visualize(results,os.path.join(IMAGE_DIR, 'validate_{}_{}_{}'.format(results['scene'], step, j)))
                        for k, v in loss_dict.items():
                            if k not in val_loss_summary_dict.keys():
                                val_loss_summary_dict[k] = []
                            val_loss_summary_dict[k].append(v)

                    loss_str = ''
                    writer_dict = {}
                    for k, v in val_loss_summary_dict.items():
                        value = np.array(v).mean()
                        loss_str += '%s: %.3f ' %(k, value)
                        writer_dict[k] = value
                    print("validation = time = %.2fm, epo %d, step %d, lr: %.5f, %ds per %d iters || loss : " % (
                        (time.time() - start) // 60,
                        epoch + 1, step + 1,
                        lr, time.time() - eval_start, print_every), loss_str)
                    loss_summary_dict = {}
                    temp = time.time()
                    writer.add_scalars('val_loss', writer_dict, step)

                trainer.train()
            step += 1
    print('===> end training')

if __name__ == '__main__':
    train()
