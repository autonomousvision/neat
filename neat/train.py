import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from config import GlobalConfig
from architectures import AttentionField
from data import CARLA_points
from utils import iou

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='default_conf', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
parser.add_argument('--val_every', type=int, default=1, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--workers', type=int, default=8, help='Dataloading workers per GPU')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')


class Engine(object):
	"""Engine that runs training and inference.
	Args
		- config: Global config.
		- cur_epoch (int): Current epoch.		
	"""

	def __init__(self, config, cur_epoch=0):
		self.cur_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e5

		self.seq_len = config.seq_len
		self.pred_len = config.pred_len
		self.tot_len = config.tot_len
		self.num_camera = config.num_camera

		self.num_class = config.num_class
		self.resolution = config.resolution
		self.offset = config.offset
		
		self.plan_points = config.plan_points
		self.plan_iters = config.plan_iters
		
		self.loss_perc = config.loss_perc
		self.loss_plan = config.loss_plan
		self.iter_losses = config.iter_losses

	def forward(self, model, data, args):
		# create batch and move to GPU
		fronts_in = data['fronts']
		lefts_in = data['lefts']
		rights_in = data['rights']

		images = []
		for i in range(self.seq_len):
			images.append(fronts_in[i].to(args.device, dtype=torch.float32))
			if self.num_camera==3:
				images.append(lefts_in[i].to(args.device, dtype=torch.float32))
				images.append(rights_in[i].to(args.device, dtype=torch.float32))

		# semantic points for network input
		query_points = data['semantic_points'].to(args.device, dtype=torch.float32)
		gt_occ = data['semantic_labels'].to(args.device)

		# target points for network input
		target_point = torch.stack(data['target_point']).to(args.device, dtype=torch.float32)

		# create driving offset label by looping over timesteps
		# label = -query + waypoint so that at test time query + label = waypoint
		gt_offsets = -query_points.clone()
		for i in range(self.tot_len):
			waypoint = torch.stack(data['waypoints'][i]).to(args.device, dtype=torch.float32)
			# create a delta tensor to add to the query points			
			delta = waypoint.transpose(0,1).unsqueeze(1) # (B, 1, 2)

			# divide to account for higher resolution
			delta = (-gt_offsets[:,:,2]==i).unsqueeze(-1) * delta / self.resolution # (B, P, 2)
			gt_offsets[:,:,:2] += delta

		gt_offsets = gt_offsets[:,:,:2].transpose(1,2) # (B, 2, P)
		gt_offsets[:,1,:] += self.offset # reconstruct only front of vehicle

		velocity = data['velocity'].to(args.device, dtype=torch.float32)

		# inference
		encoding = model.encoder(images, velocity)
		pred_occ, pred_off, _ = model.decode(query_points, target_point, encoding)
		
		return pred_occ, pred_off, gt_occ, gt_offsets

	def train(self, optimizer, model, args, dataloader_train):
		loss_epoch = 0.
		model.train()

		# train loop
		for batch_num, data in enumerate(tqdm(dataloader_train), 0):
			
			# efficiently zero gradients
			for p in model.parameters():
				p.grad = None
			
			pred_occ, pred_off, gt_occ, gt_offsets = self.forward(model, data, args)

			# losses over iterative predictions
			loss = 0.
			for i, pred_occ_i in enumerate(pred_occ):
				loss += self.iter_losses[i] * self.loss_perc * F.cross_entropy(pred_occ_i, gt_occ).mean()
			for i, pred_off_i in enumerate(pred_off):
				loss += self.iter_losses[i] * self.loss_plan * F.l1_loss(pred_off_i, gt_offsets).mean()
			loss.backward()
			loss_epoch += float(loss.item())

			optimizer.step()
		
		loss_epoch = loss_epoch / batch_num
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1

	def validate(self, model, args, dataloader_val):
		model.eval()

		with torch.no_grad():	
			intersection_epoch = [0.] * self.num_class
			union_epoch = [0.] * self.num_class
			off_epoch = 0.

			# validation loop
			for batch_num, data in enumerate(tqdm(dataloader_val), 0):
				
				pred_occ, pred_off, gt_occ, gt_offsets = self.forward(model, data, args)
				pred_occ_class = torch.argmax(pred_occ[-1], dim=1)

				# losses
				for k in range(self.num_class):
					gt_occ_k = gt_occ==k
					pred_occ_k = pred_occ_class==k
					for pt1, pt2 in zip(gt_occ_k, pred_occ_k):
						intersection, union = iou(pt1, pt2)
						intersection_epoch[k] += float(intersection.item())
						union_epoch[k] += float(union.item())

				off_epoch += float(F.l1_loss(pred_off[-1], gt_offsets).mean())

			out_loss = np.array(intersection_epoch) / np.array(union_epoch)
			off_loss = off_epoch / float(batch_num)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}: Off: {off_loss:3.3f}')
			for k in range(self.num_class):
				tqdm.write(f'Class {k:02d}: IoU: {out_loss[k]:3.3f}')

			self.val_loss.append(off_loss)

	def save(self, optimizer, model, args):

		save_best = False
		if self.val_loss[-1] <= self.bestval:
			self.bestval = self.val_loss[-1]
			save_best = True
		
		# create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'bestval': self.bestval,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss,
		}

		# save the recent model/optimizer states
		torch.save(model.encoder.state_dict(), os.path.join(args.logdir, 'encoder.pth'))
		torch.save(model.decoder.state_dict(), os.path.join(args.logdir, 'decoder.pth'))

		torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

		# log other data corresponding to the recent model
		with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
			f.write(json.dumps(log_table))

		tqdm.write('====== Saved recent model ======>')
		
		if save_best:
			torch.save(model.encoder.state_dict(), os.path.join(args.logdir, 'encoder_'+str(self.cur_epoch)+'.pth'))
			torch.save(model.decoder.state_dict(), os.path.join(args.logdir, 'decoder_'+str(self.cur_epoch)+'.pth'))

			torch.save(model.encoder.state_dict(), os.path.join(args.logdir, 'best_encoder.pth'))
			torch.save(model.decoder.state_dict(), os.path.join(args.logdir, 'best_decoder.pth'))

			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')


def main():
	# get args
	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	# create logdir
	if not os.path.isdir(args.logdir):
		os.makedirs(args.logdir)
		print('Created dir:', args.logdir)

	# config
	conf = GlobalConfig()
	
	# datasets
	train_set = CARLA_points(conf.train_data, conf)
	val_set = CARLA_points(conf.val_data, conf)

	# dataloaders
	dataloader_train = DataLoader(train_set, batch_size=args.batch_size,
									shuffle=True, num_workers=args.workers, pin_memory=True)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size,
									shuffle=False, num_workers=args.workers, pin_memory=True)

	# model
	model = AttentionField(conf, args.device)

	parameters = list(model.encoder.parameters()) + list(model.decoder.parameters())

	optimizer = optim.AdamW(parameters, lr=conf.lr)
	trainer = Engine(conf)

	if os.path.isdir(args.logdir):
		logfile = os.path.join(args.logdir, 'recent.log')
		if os.path.isfile(logfile):
			print('Loading checkpoint from ' + args.logdir)
			with open(logfile, 'r') as f:
				log_table = json.load(f)

			# load variables
			trainer.cur_epoch = log_table['epoch']
			trainer.bestval = log_table['bestval']
			trainer.train_loss = log_table['train_loss']
			trainer.val_loss = log_table['val_loss']

			# load checkpoint
			model.encoder.load_state_dict(torch.load(os.path.join(args.logdir, 'encoder.pth')))
			model.decoder.load_state_dict(torch.load(os.path.join(args.logdir, 'decoder.pth')))

			optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'recent_optim.pth')))

	# log args
	with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
		json.dump(args.__dict__, f, indent=2)

	for epoch in range(trainer.cur_epoch, args.epochs):
		trainer.train(optimizer, model, args, dataloader_train)
		if epoch % args.val_every == 0: 
			trainer.validate(model, args, dataloader_val)
			trainer.save(optimizer, model, args)


if __name__ == '__main__':
    main()