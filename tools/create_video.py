from PIL import Image, ImageFont, ImageDraw
import csv
import os
import sys
import json
import numpy as np
import skvideo.io
import multiprocessing

def process_fn(episode):
	writer = skvideo.io.FFmpegWriter(os.path.join(sys.argv[2], episode.split('/')[-1]+'_'+sys.argv[3].replace("/", "_")+'.mp4'), inputdict={'-r':sys.argv[4]}, 
									 outputdict={'-r':sys.argv[4], '-acodec':'aac', '-c:v': 'libx264', '-preset': 'slow', '-pix_fmt': 'yuv420p'})
	print ('path: ', os.path.join(sys.argv[1], episode, sys.argv[3]))
	for image_path in sorted(os.listdir(os.path.join(sys.argv[1], episode, sys.argv[3]))):
		img = Image.open(os.path.join(sys.argv[1], episode, sys.argv[3], image_path))
		d = ImageDraw.Draw(img)
		writer.writeFrame(np.asanyarray(img).copy())

	writer.close()

if __name__ == '__main__':

	if not os.path.isdir(os.path.join(sys.argv[2])):
		os.mkdir(os.path.join(sys.argv[2]))

	episodes = sorted(os.listdir(sys.argv[1]))

	jobs = []
	cnt = 0
	for episode in episodes:
		# if cnt == 20:
		# 	break
		cnt += 1
		if os.path.isdir(os.path.join(sys.argv[1], episode)):
			p = multiprocessing.Process(target=process_fn, args=(episode,))
			jobs.append(p)
			p.start()


	for process in jobs:
		process.join()
