import argparse
import json
import os
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from config import GlobalConfig
from model import TransFuser
from data import CARLA_Data

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='transfuser', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=101, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--sensor-failures','-sf', nargs='+', type=str, choices=['rgb','lidar'], default=None, help='If sensor is added here, it will be randomly dropped during some epochs during training to make it more robust against sensor failure')
parser.add_argument('--prob','-p', type=float, default=0.3, help='probability with which to drop a sensor')
parser.add_argument('--use-classification-branch', '-c', action='store_true')

args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)

writer = SummaryWriter(log_dir=args.logdir)


class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self,  cur_epoch=0, cur_iter=0):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e10
		self.sensor_failures = [] if args.sensor_failures == None else args.sensor_failures
		self.sensor_failures.append('None')
		print(f'Using sensor corruption modes:{self.sensor_failures}')
		self.use_classification_branch = False if args.use_classification_branch == None else args.use_classification_branch
		

	def train(self):
		loss_epoch = 0.
		num_batches = 0
		model.train()

		# Train loop
		for data in tqdm(dataloader_train):
			
			# efficiently zero gradients
			for p in model.parameters():
				p.grad = None
			
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
			rgb_corrupted = False
			lidar_corrupted = False
			current_batch_size = fronts_in[0].shape[0]
			for i in range(config.seq_len):
				self.current_sensor_failure_used = random.choice(self.sensor_failures)
				gt_front = torch.zeros(current_batch_size, dtype=torch.float32)
				if self.current_sensor_failure_used == 'rgb' :
					possible_fronts = [fronts_in[i], torch.zeros_like(fronts_in[i])]
					# front = random.choices(possible_fronts, weights=[1-args.prob, args.prob])[0]
					front = torch.zeros_like(fronts_in[i])
					if torch.max(front) == 0:
						rgb_corrupted = True
						gt_front = torch.ones(current_batch_size, dtype=torch.float32)
					fronts.append(front.to(args.device, dtype=torch.float32))
				else:
					fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
				if not config.ignore_sides:
					lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
					rights.append(rights_in[i].to(args.device, dtype=torch.float32))
				if not config.ignore_rear:
					rears.append(rears_in[i].to(args.device, dtype=torch.float32))
				gt_lidar = torch.zeros(current_batch_size, dtype=torch.float32)
				if self.current_sensor_failure_used == 'lidar' and not rgb_corrupted:
					possible_lidars = [lidars_in[i], torch.zeros_like(lidars_in[i])]
					# lidar = np.random.choice(possible_lidars, 1, [1-args.prob, args.prob]).tolist()[0]
					lidar = random.choices(possible_lidars, weights=[1-args.prob, args.prob])[0]
					lidar = torch.zeros_like(lidars_in[i])
					lidars.append(lidar.to(args.device, dtype=torch.float32))

					if lidar.mean() == 0:
						lidar_corrupted = True
						gt_lidar = torch.ones(current_batch_size, dtype=torch.float32)
				else:
					lidars.append(lidars_in[i].to(args.device, dtype=torch.float32))

			gt_both_working = torch.zeros(current_batch_size, dtype=torch.float32) if lidar_corrupted or rgb_corrupted else torch.ones(current_batch_size, dtype=torch.float32)
			# driving labels
			command = data['command'].to(args.device)
			gt_velocity = data['velocity'].to(args.device, dtype=torch.float32)
			gt_steer = data['steer'].to(args.device, dtype=torch.float32)
			gt_throttle = data['throttle'].to(args.device, dtype=torch.float32)
			gt_brake = data['brake'].to(args.device, dtype=torch.float32)

			# target point
			target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)
			
			if self.use_classification_branch:
				pred_wp, logits = model(fronts+lefts+rights+rears, lidars, target_point, gt_velocity)
			else:
				pred_wp = model(fronts+lefts+rights+rears, lidars, target_point, gt_velocity)
			


			if self.use_classification_branch:
				gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(config.seq_len, len(data['waypoints']))]
				gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
				gt_classification = torch.stack([gt_front, gt_lidar, gt_both_working], dim=1).to(args.device, dtype=torch.float32)
				loss = F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean()
				classification_loss = F.cross_entropy(logits, gt_classification)
				# print(f'Loss {loss.item()}')
				# print(f'Classification Loss {classification_loss.item()}')

				total_loss = sum([loss, classification_loss])
				total_loss.backward()
				loss_epoch += float(total_loss.item())
				# wp_loss_epoch += float(total_loss.item())
				# classification_loss_epoch += float(total_loss.item())

			else:
				gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(config.seq_len, len(data['waypoints']))]
				gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
				total_loss = F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean()
				total_loss.backward()
				loss_epoch += float(total_loss.item())


			num_batches += 1
			optimizer.step()

			writer.add_scalar('train_loss', total_loss.item(), self.cur_iter)
			self.cur_iter += 1
		
		
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1

	def validate(self):
		model.eval()

		with torch.no_grad():	
			num_batches = 0
			wp_epoch = 0.
			classification_loss_epoch = 0.

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
				current_batch_size = fronts_in[0].shape[0]
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
				gt_lidar = torch.zeros(current_batch_size, dtype=torch.float32)
				gt_front = torch.zeros(current_batch_size, dtype=torch.float32)
				gt_both_working = torch.ones(current_batch_size, dtype=torch.float32)

				# target point
				target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)

				if self.use_classification_branch:
					pred_wp, logits = model(fronts+lefts+rights+rears, lidars, target_point, gt_velocity)
				else:
					pred_wp = model(fronts+lefts+rights+rears, lidars, target_point, gt_velocity, train=False)

				if self.use_classification_branch:
					gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(config.seq_len, len(data['waypoints']))]
					gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
					gt_classification = torch.stack([gt_front, gt_lidar, gt_both_working], dim=1).to(args.device, dtype=torch.float32)
					loss = F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean()
					classification_loss = F.cross_entropy(logits, gt_classification)
					total_loss = sum([loss, classification_loss])
					wp_epoch += float(total_loss.item())
					classification_loss_epoch += float(classification_loss.item())
				# wp_loss_epoch += float(total_loss.item())
				# classification_loss_epoch += float(total_loss.item())
				else:
					gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(config.seq_len, len(data['waypoints']))]
					gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
					wp_epoch += float(F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean())

				num_batches += 1
					
			wp_loss = wp_epoch / float(num_batches)
			classification_loss_final = classification_loss_epoch / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {wp_loss:3.3f}')

			writer.add_scalar('val_loss', wp_loss, self.cur_epoch)
			writer.add_scalar('val_class_loss', classification_loss_final, self.cur_epoch)
			
			self.val_loss.append(wp_loss)

	def save(self):

		save_best = False
		if self.val_loss[-1] <= self.bestval:
			self.bestval = self.val_loss[-1]
			self.bestval_epoch = self.cur_epoch
			save_best = True
		
		# Create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'iter': self.cur_iter,
			'bestval': self.bestval,
			'bestval_epoch': self.bestval_epoch,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss,
		}

		# Save ckpt for every epoch
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth'%self.cur_epoch))

		# Save the recent model/optimizer states
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
		torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

		# Log other data corresponding to the recent model
		with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
			f.write(json.dumps(log_table))

		tqdm.write('====== Saved recent model ======>')
		
		if save_best:
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')

# Config
config = GlobalConfig()

# Data
train_set = CARLA_Data(root=config.train_data, config=config)
val_set = CARLA_Data(root=config.val_data, config=config)

dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Model
model = TransFuser(config, args.device)
if args.use_classification_branch:
	print(args.use_classification_branch)
	model = TransFuser(config, args.device, args.use_classification_branch)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)
trainer = Engine()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total trainable parameters: ', params)

# Create logdir
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)
	print('Created dir:', args.logdir)
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
	print('Loading checkpoint from ' + args.logdir)
	with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
		log_table = json.load(f)

	# Load variables
	trainer.cur_epoch = log_table['epoch']
	if 'iter' in log_table: trainer.cur_iter = log_table['iter']
	trainer.bestval = log_table['bestval']
	trainer.train_loss = log_table['train_loss']
	trainer.val_loss = log_table['val_loss']

	# Load checkpoint
	model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
	optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'recent_optim.pth')))

# Log args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
	json.dump(args.__dict__, f, indent=2)

for epoch in range(trainer.cur_epoch, args.epochs): 
	trainer.train()
	if epoch % args.val_every == 0: 
		trainer.validate()
		trainer.save()