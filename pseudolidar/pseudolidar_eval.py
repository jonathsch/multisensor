from pseudo_lidar_dataset import PseudoLidarData
from transfuser.model import TransFuser
from transfuser.config import GlobalConfig
import argparse
import json
import os
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, required=True, help='Weights path for evaluation.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size')

args = parser.parse_args()


class Engine(object):
    """Engine that runs training and inference.
    Args
            - cur_epoch (int): Current epoch.
            - print_every (int): How frequently (# batches) to print loss.
            - validate_every (int): How frequently (# epochs) to run validation.

    """

    def __init__(self):
        self.val_loss = []

    def validate(self):
        model.eval()

        with torch.no_grad():
            num_batches = 0
            wp_epoch = 0.

            # Validation loop
            for batch_num, data in enumerate(tqdm(dataloader_val), 0):

                # create batch and move to GPU
                fronts_in = data['fronts']
                lefts_in = data['lefts']
                rights_in = data['rights']
                rears_in = data['rears']
                lidars_in = data['lidars']
                fronts = []
                lefts = []
                rights = []
                rears = []
                lidars = []
                for i in range(config.seq_len):
                    fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
                    if not config.ignore_sides:
                        lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
                        rights.append(rights_in[i].to(args.device, dtype=torch.float32))
                    if not config.ignore_rear:
                        rears.append(rears_in[i].to(args.device, dtype=torch.float32))
                    lidars.append(lidars_in[i].to(args.device, dtype=torch.float32))

                # driving labels
                command = data['command'].to(args.device)
                gt_velocity = data['velocity'].to(args.device, dtype=torch.float32)
                gt_steer = data['steer'].to(args.device, dtype=torch.float32)
                gt_throttle = data['throttle'].to(args.device, dtype=torch.float32)
                gt_brake = data['brake'].to(args.device, dtype=torch.float32)

                # target point
                target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)

                pred_wp = model(fronts+lefts+rights+rears, lidars, target_point, gt_velocity)

                gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32)
                                for i in range(config.seq_len, len(data['waypoints']))]
                gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
                wp_epoch += float(F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean())

                num_batches += 1

            wp_loss = wp_epoch / float(num_batches)
            tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {wp_loss:3.3f}')


# Config
config = GlobalConfig()

# Data
val_set = PseudoLidarData(root=config.val_data, config=config)
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Model
model = TransFuser(config, args.device)
model.load_state_dict(torch.load(args.weights))
trainer = Engine()

trainer.validate()
