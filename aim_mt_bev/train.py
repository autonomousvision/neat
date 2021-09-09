import argparse
import json
import os
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from architectures import MultiTaskImageBDNetwork
from data import CARLA_points
from config import GlobalConfig


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='aim_bev', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=101, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=72, help='Batch size')
parser.add_argument('--seq_len', type=int, default=1, help='Input sequence length (factor of 10)')
parser.add_argument('--pred_len', type=int, default=4, help='number of timesteps to predict')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)


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
		self.bestval = -1e5

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

			fronts = []
			lefts = []
			rights = []

			for i in range(args.seq_len):
				fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
				if not config.ignore_sides:
					lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
					rights.append(rights_in[i].to(args.device, dtype=torch.float32))

			# driving labels
			command = data['command'].to(args.device)
			gt_velocity = data['velocity'].to(args.device, dtype=torch.float32)
			gt_steer = data['steer'].to(args.device, dtype=torch.float32)
			gt_throttle = data['throttle'].to(args.device, dtype=torch.float32)
			gt_brake = data['brake'].to(args.device, dtype=torch.float32)

			# target point
			target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)

			# inference
			encoding = [model.image_encoder(fronts)]
			if not config.ignore_sides:
				encoding.append(model.image_encoder(lefts))
				encoding.append(model.image_encoder(rights))

			# encode velocity
			# encoding.append(model.velocity_encoder(gt_velocity.unsqueeze(1)))

			pred_sem = model.sem_decoder(encoding)
			gt_sem = data['topdowns'][0].to(args.device, dtype=torch.long)
			gt_sem = gt_sem[:,:,:config.axis,config.offset:config.axis+config.offset].squeeze(1)

			pred_wp = model(encoding, target_point)
			
			gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(args.seq_len, len(data['waypoints']))]
			gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
			wp_loss = F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean()
			sem_loss = F.cross_entropy(pred_sem,gt_sem).mean()
			loss = wp_loss + config.loss_perc * sem_loss
			loss.backward()
			loss_epoch += float(wp_loss.item())

			num_batches += 1
			optimizer.step()

			self.cur_iter += 1
		
		
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1

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
				fronts = []
				lefts = []
				rights = []

				for i in range(args.seq_len):
					fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
					if not config.ignore_sides:
						lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
						rights.append(rights_in[i].to(args.device, dtype=torch.float32))

				# driving labels
				command = data['command'].to(args.device)
				gt_velocity = data['velocity'].to(args.device, dtype=torch.float32)
				gt_steer = data['steer'].to(args.device, dtype=torch.float32)
				gt_throttle = data['throttle'].to(args.device, dtype=torch.float32)
				gt_brake = data['brake'].to(args.device, dtype=torch.float32)

				# target point
				target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)

				# inference
				encoding = [model.image_encoder(fronts)]
				if not config.ignore_sides:
					encoding.append(model.image_encoder(lefts))
					encoding.append(model.image_encoder(rights))

				# encode velocity
				# encoding.append(model.velocity_encoder(gt_velocity.unsqueeze(1)))

				pred_wp = model(encoding, target_point)

				gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(args.seq_len, len(data['waypoints']))]
				gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
				wp_epoch += float(F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean())

				num_batches += 1
					
			wp_loss = wp_epoch / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {wp_loss:3.3f}')

			self.val_loss.append(1.0 - wp_loss)

	def save(self):

		save_best = False
		if self.val_loss[-1] >= self.bestval:
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
train_set = CARLA_points(root=config.train_data, config=config)
val_set = CARLA_points(root=config.val_data, config=config)

dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Model
model = MultiTaskImageBDNetwork(config, args.device)
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