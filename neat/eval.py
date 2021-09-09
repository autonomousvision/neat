import argparse
import json
import os
import sys
from tqdm import tqdm
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from config import GlobalConfig
from architectures import AttentionField
from data import CARLA_points
from utils import iou, flow_to_color


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--vis', action='store_true', help='Visualize each model while evaluating')
parser.add_argument('--vis_freq', type=int, default=100, help='Visualization frequency')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--out_res', type=int, default=256, help='output image resolution')
args = parser.parse_args()

# config
conf = GlobalConfig()

# data
val_set = CARLA_points(conf.val_data, conf)
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# model
model = AttentionField(conf, args.device)

# load saved weights
model.encoder.load_state_dict(torch.load('log/{}/best_encoder.pth'.format(args.id)))
model.decoder.load_state_dict(torch.load('log/{}/best_decoder.pth'.format(args.id)))

# image storage directories
if args.vis:
	if not os.path.isdir(f"log/{args.id}/img"):
		os.makedirs(f"log/{args.id}/img")
	if not os.path.isdir(f"log/{args.id}/sem"):
		os.makedirs(f"log/{args.id}/sem")
	if not os.path.isdir(f"log/{args.id}/out"):
		os.makedirs(f"log/{args.id}/out")
	if not os.path.isdir(f"log/{args.id}/flow"):
		os.makedirs(f"log/{args.id}/flow")

intersection_epoch = [0.] * conf.num_class
union_epoch = [0.] * conf.num_class
off_epoch = 0.
wp_epoch = 0.
match = 0
miss = 0
fp = 0
converter = np.uint8(conf.converter) # used for semantics

with torch.no_grad():
	model.eval()

	for batch_num, data in enumerate(tqdm(dataloader_val), 0):
		
		# create batch and move to GPU
		fronts_in = data['fronts']
		lefts_in = data['lefts']
		rights_in = data['rights']

		images = []
		for i in range(conf.seq_len):
			images.append(fronts_in[i].to(args.device, dtype=torch.float32))
			if conf.num_camera==3:
				images.append(lefts_in[i].to(args.device, dtype=torch.float32))
				images.append(rights_in[i].to(args.device, dtype=torch.float32))
		
		# semantic points for network input
		query_points = data['semantic_points'].to(args.device, dtype=torch.float32)
		gt_occ = data['semantic_labels'].to(args.device)

		# target points for network input
		target_point = torch.stack(data['target_point']).to(args.device, dtype=torch.float32)
		
		# waypoints for visualization
		waypoints = []
		
		# create driving offset label by looping over timesteps
		# label = -query + waypoint so that at test time query + label = waypoint
		gt_offsets = -query_points.clone()
		for i in range(conf.tot_len):
			waypoint = torch.stack(data['waypoints'][i]).to(args.device, dtype=torch.float32)
			waypoints.append(waypoint)

			# create a delta tensor to add to the query points			
			delta = waypoint.transpose(0,1).unsqueeze(1) # (B, 1, 2)

			# divide to account for higher resolution
			delta = (-gt_offsets[:,:,2]==i).unsqueeze(-1) * delta / conf.resolution # (B, P, 2)
			gt_offsets[:,:,:2] += delta

		gt_offsets = gt_offsets[:,:,:2].transpose(1,2) # (B, 2, P)
		gt_offsets[:,1,:] += conf.offset # reconstruct only front of vehicle

		velocity = data['velocity'].to(args.device, dtype=torch.float32)

		# inference
		encoding = model.encoder(images, velocity)
		pred_occ, pred_off, _ = model.decode(query_points, target_point, encoding)

		# waypoint prediction
		pred_waypoint_mean, red_light_occ = model.plan(target_point, encoding, conf.plan_scale, conf.plan_points, conf.plan_iters)
		
		wp_pred = pred_waypoint_mean[:,conf.seq_len:]
		wp_gt = torch.stack(waypoints[conf.seq_len:], dim=1).transpose(0,2)

		# s,t,b = model.control_pid(wp_pred, velocity, target_point, red_light_occ)

		# grid used for visualizing occupancy and flow
		linspace_x = torch.linspace(-conf.axis/2, conf.axis/2, steps=args.out_res)
		linspace_y = torch.linspace(-conf.axis/2, conf.axis/2, steps=args.out_res)
		linspace_t = torch.linspace(0, conf.tot_len - 1, steps=conf.tot_len)
		
		# gt semantics
		semantics = (data['topdowns'][0][0][0].data.cpu().numpy()).astype(np.uint8)
		semantics = converter[semantics][:conf.axis,conf.offset:conf.axis+conf.offset]
		red_light_gt = (semantics==3).sum()

		if red_light_gt and red_light_occ:
			match += 1
		if red_light_gt and red_light_occ==0:
			miss += 1
		if red_light_gt==0 and red_light_occ:
			fp += 1

		if args.vis and (batch_num % args.vis_freq == 0):
			
			for i in range(conf.seq_len):
				# save one sample per batch
				if not os.path.isdir(f"log/{args.id}/img/{str(i)}"):
					os.makedirs(f"log/{args.id}/img/{str(i)}")
				front_numpy = (fronts_in[i][0].data.cpu().numpy().transpose((1, 2, 0))).astype(np.uint8)
				left_numpy = (lefts_in[i][0].data.cpu().numpy().transpose((1, 2, 0))).astype(np.uint8)
				right_numpy = (rights_in[i][0].data.cpu().numpy().transpose((1, 2, 0))).astype(np.uint8)
				image_numpy = np.concatenate([left_numpy,front_numpy,right_numpy], axis=1)
				image_display = Image.fromarray(image_numpy)
				image_display.save(f"log/{args.id}/img/{str(i)}/{str(batch_num).zfill(4)}.png")

			# target point in pixel coordinates
			target_point_pixel = target_point.squeeze().cpu().numpy()
			target_point_pixel[1] += conf.offset * conf.resolution

			# hack for when actual target is outside image (axis/2 * resolution)
			target_point_pixel = np.clip(target_point_pixel, -(conf.axis/2 * conf.resolution - 1), (conf.axis/2 * conf.resolution - 1)) 
			target_point_pixel = (target_point_pixel*args.out_res//50 + args.out_res//2).astype(np.uint8)

			for i in range(conf.tot_len):
				if not os.path.isdir(f"log/{args.id}/sem/{str(i)}"):
					os.makedirs(f"log/{args.id}/sem/{str(i)}")
				if not os.path.isdir(f"log/{args.id}/out/{str(i)}"):
					os.makedirs(f"log/{args.id}/out/{str(i)}")
				if not os.path.isdir(f"log/{args.id}/flow/{str(i)}"):
					os.makedirs(f"log/{args.id}/flow/{str(i)}")
					
				# gt semantics
				semantics = (data['topdowns'][i][0][0].data.cpu().numpy()).astype(np.uint8)
				semantics = converter[semantics][:conf.axis,conf.offset:conf.axis+conf.offset]
				semantic_display = np.zeros((semantics.shape[0], semantics.shape[1], 3))
				for key, value in conf.classes.items():
					semantic_display[np.where(semantics == key)] = value
				semantic_display = semantic_display.astype(np.uint8)
				semantic_display = Image.fromarray(semantic_display)
				semantic_display.save(f"log/{args.id}/sem/{str(i)}/{str(batch_num).zfill(4)}.png")

				# gt waypoint in pixel coordinates
				img_waypoint = waypoints[i].data.cpu().numpy()
				img_waypoint[1] += conf.offset * conf.resolution
				img_waypoint = np.clip(img_waypoint, -(conf.axis/2 * conf.resolution - 1), (conf.axis/2 * conf.resolution - 1)) 
				img_waypoint = (img_waypoint*args.out_res//(conf.axis * conf.resolution) + args.out_res//2).astype(np.uint8)

				# predicted waypoint in pixel coordinates
				pred_waypoint = pred_waypoint_mean[0,i].data.cpu().numpy()
				pred_waypoint[1] += conf.offset * conf.resolution
				pred_waypoint = np.clip(pred_waypoint, -(conf.axis/2 * conf.resolution - 1), (conf.axis/2 * conf.resolution - 1)) 
				pred_waypoint = (pred_waypoint*args.out_res//(conf.axis * conf.resolution) + args.out_res//2).astype(np.uint8)
				
				# visualization of occupancy and flow
				img_rows = []
				flow_rows = []
				for row in range(args.out_res):
					grid_x, grid_y, grid_t = torch.meshgrid(linspace_x, linspace_y[row], linspace_t[i].unsqueeze(0))
					grid_points = torch.stack((grid_x, grid_y, grid_t), dim=3).unsqueeze(0).repeat(args.batch_size,1,1,1,1)
					grid_points = grid_points.reshape(args.batch_size,-1,3).to(args.device, dtype=torch.float32)
					pred_img_pts, pred_img_offsets, _ = model.decode(grid_points, target_point, encoding)
					pred_img_pts = torch.argmax(pred_img_pts[-1], dim=1)
					pred_img = pred_img_pts.reshape(args.batch_size,args.out_res)
					pred_flow = pred_img_offsets[-1].reshape(args.batch_size,2,args.out_res)
					img_rows.append(pred_img)
					flow_rows.append(pred_flow)
				
				pred_img = torch.stack(img_rows, dim=-1)
				pred_flow = torch.stack(flow_rows, dim=-1)

				semantics = pred_img[0,:,:].transpose(1, 0).data.cpu().numpy().astype(np.uint8)
				semantic_display = np.zeros((semantics.shape[0], semantics.shape[1], 3))
				for key, value in conf.classes.items():
					semantic_display[np.where(semantics == key)] = value
				semantic_display = semantic_display.astype(np.uint8)
				semantic_display = Image.fromarray(semantic_display)

				semantic_display.save(f"log/{args.id}/out/{str(i)}/{str(batch_num).zfill(4)}.png")

				# flow image of predicted offsets
				flow_uv = pred_flow[0,:,:,:].transpose(2,0).data.cpu().numpy()*args.out_res/conf.axis
				flow_rgb = flow_to_color(flow_uv)

				flow_display = Image.fromarray(flow_rgb)
				draw = ImageDraw.Draw(flow_display)
				draw.ellipse([tuple(target_point_pixel-2), tuple(target_point_pixel+2)], fill='Blue', outline='Blue')
				draw.ellipse([tuple(img_waypoint-2), tuple(img_waypoint+2)], fill='Green', outline='Green')
				draw.ellipse([tuple(pred_waypoint-2), tuple(pred_waypoint+2)], fill='Red', outline='Red')
				flow_display.save(f"log/{args.id}/flow/{str(i)}/{str(batch_num).zfill(4)}.png")

		pred_occ_class = torch.argmax(pred_occ[-1], dim=1)
		# losses
		for k in range(conf.num_class):
			gt_occ_k = gt_occ==k
			pred_occ_k = pred_occ_class==k
			for pt1, pt2 in zip(gt_occ_k, pred_occ_k):
				intersection, union = iou(pt1, pt2)
				intersection_epoch[k] += float(intersection.item())
				union_epoch[k] += float(union.item())

		off_epoch += float(F.l1_loss(pred_off[-1], gt_offsets).mean())
		wp_epoch += float(F.l1_loss(wp_gt,wp_pred).mean())

out_loss = np.array(intersection_epoch) / np.array(union_epoch)
off_loss = off_epoch / float(batch_num)
wp_loss = wp_epoch / float(batch_num)
print (f'Off: {off_loss:3.3f}')
print (f'Wp: {wp_loss:3.3f}')
print (f'Match: {match}')
print (f'Miss: {miss}')
print (f'FP: {fp}')
for k in range(conf.num_class):
	print(f'Class {k:02d}: IoU: {out_loss[k]:3.3f}')