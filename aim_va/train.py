import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from model import MultiTaskImageNetwork
from data import CARLA_Data
from class_converter import sub_classes


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='aim_vis_abs', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
parser.add_argument('--ignore_sides', action='store_true', help='Ignores side cameras')
parser.add_argument('--ignore_rear', action='store_true', help='Ignores rear camera')
parser.add_argument('--classes', type=str, default='no_stop')
parser.add_argument('--seq_len', type=int, default=1, help='Input sequence length (factor of 10)')
parser.add_argument('--pred_len', type=int, default=4, help='number of timesteps to predict')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--input_scale', type=int, default=1, help='Inverse input scale factor')
parser.add_argument('--input_crop', type=float, default=0.64, help='Input crop size')
args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)


class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self, conf_log, cur_epoch=0, cur_iter=0):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = -1e5

	def train(self):
		loss_epoch = 0.
		wp_epoch = 0.
		num_batches = 0
		sep_wp_loss = torch.zeros(args.pred_len).to(args.device, dtype=torch.float32)

		model.train()

		# Train loop
		for data in dataloader_train:
			
			# efficiently zero gradients
			for p in model.parameters():
				p.grad = None
			
			# create batch and move to GPU
			fronts_in = data['fronts']
			fronts = []
			if not args.ignore_sides:
				lefts_in = data['lefts']
				rights_in = data['rights']
				lefts = []
				rights = []
			if not args.ignore_rear:
				rears_in = data['rears']
				rears = []
			for i in range(args.seq_len):
				fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
				if not args.ignore_sides:
					lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
					rights.append(rights_in[i].to(args.device, dtype=torch.float32))
				if not args.ignore_rear:
					rears.append(rears_in[i].to(args.device, dtype=torch.float32))

			# target point
			target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)

			# inference
			encoding = [model.image_encoder(fronts)]
			if not args.ignore_sides:
				encoding.append(model.image_encoder(lefts))
				encoding.append(model.image_encoder(rights))
			if not args.ignore_rear:
				encoding.append(model.image_encoder(rears))

			pred_wp = model(encoding, target_point)
			
			gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(args.seq_len, len(data['waypoints']))]
			gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)

			loss = F.l1_loss(pred_wp, gt_waypoints, reduction='none')
			sep_wp_loss += loss.mean((0,2))
			
			loss.mean().backward()
			wp_epoch += loss.mean().item() 

			num_batches += 1
			optimizer.step()
			
			self.cur_iter += 1
		
		
		loss_epoch = wp_epoch / num_batches
		sep_wp_loss = sep_wp_loss / num_batches

		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1
		

	def validate(self):
		model.eval()

		with torch.no_grad():	
			num_batches = 0
			wp_epoch = 0.
			sep_wp_loss = torch.zeros(args.pred_len).to(args.device, dtype=torch.float32)

			wp_loss_list = [[]]

			# Validation loop
			for data in dataloader_val:
				
				# create batch and move to GPU
				fronts_in = data['fronts']
				fronts = []
				if not args.ignore_sides:
					lefts_in = data['lefts']
					rights_in = data['rights']
					lefts = []
					rights = []
				if not args.ignore_rear:
					rears_in = data['rears']
					rears = []
				for i in range(args.seq_len):
					fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
					if not args.ignore_sides:
						lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
						rights.append(rights_in[i].to(args.device, dtype=torch.float32))
					if not args.ignore_rear:
						rears.append(rears_in[i].to(args.device, dtype=torch.float32))

				# target point
				target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)

				# inference
				encoding = [model.image_encoder(fronts)]
				if not args.ignore_sides:
					encoding.append(model.image_encoder(lefts))
					encoding.append(model.image_encoder(rights))
				if not args.ignore_rear:
					encoding.append(model.image_encoder(rears))

				pred_wp = model(encoding, target_point)

				gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(args.seq_len, len(data['waypoints']))]
				gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
				loss = F.l1_loss(pred_wp, gt_waypoints, reduction='none')
				wp_epoch += loss.mean().item()
				sep_wp_loss += loss.mean((0,2))
				num_batches += 1

				sep_item_loss = loss.mean((1,2)).detach().cpu().numpy()

				for i, _loss in enumerate(sep_item_loss):

					wp_loss_list[0].append(_loss)

			wp_loss = wp_epoch / num_batches
			sep_wp_loss = sep_wp_loss / num_batches

			print(f'Epoch {self.cur_epoch:03d}, ' + f' Wp: {wp_loss:3.3f}')
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

		torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
		torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

		# Log other data corresponding to the recent model
		with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
			f.write(json.dumps(log_table))

		print('====== Saved recent model ======>')

		
		if save_best:
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			print('====== Overwrote best model ======>')


# Data
root_dir = '/is/rg/avg/kchitta/carla9-10_data/2021/apv3'
train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10']
val_towns = ['Town01_long', 'Town02_long', 'Town03_long', 'Town04_long', 'Town05_long', 'Town06_long']
train_data, val_data = [], []
for town in train_towns:
	train_data.append(os.path.join(root_dir, town))
	train_data.append(os.path.join(root_dir, town+'_small'))
for town in val_towns:
	val_data.append(os.path.join(root_dir, town))

class_converter = sub_classes[args.classes]
print("classes: ", class_converter)

train_set = CARLA_Data(root=train_data, 
	pred_len=args.pred_len, 
	class_converter=class_converter, 
	ignore_sides=args.ignore_sides, 
	ignore_rear=args.ignore_rear,
	seq_len=args.seq_len,
	input_scale=args.input_scale,
	input_crop=args.input_crop)
val_set = CARLA_Data(root=val_data, 
	pred_len=args.pred_len, 
	class_converter=class_converter, 
	ignore_sides=args.ignore_sides, 
	ignore_rear=args.ignore_rear,
	seq_len=args.seq_len,
	input_scale=args.input_scale,
	input_crop=args.input_crop)

dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True) 
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True) 

# Model
num_segmentation_classes = len(np.unique(class_converter))

num_cameras = 1
if not args.ignore_sides:
	num_cameras += 2
if not args.ignore_rear:
	num_cameras += 1
model = MultiTaskImageNetwork('cuda', num_segmentation_classes, args.pred_len, num_cameras)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
conf_log = {
	"id": args.id,
	"epochs": args.epochs,
	"batch_size": args.batch_size,
	"lr": args.lr,
	"seq_len": args.seq_len,
	"pred_len": args.pred_len,
	"classes": class_converter,
	"class_name": args.classes,
	"num_cameras": num_cameras,
    }
trainer = Engine(conf_log)

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