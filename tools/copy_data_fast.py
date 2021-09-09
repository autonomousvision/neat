import os
import sys
import time
import shutil
import multiprocessing
import argparse


def copy_episode(episode):
	print (episode)
	if not os.path.isdir(os.path.join(args.target_dir, episode)):
		os.mkdir(os.path.join(args.target_dir, episode))

	episode_data = sorted(os.listdir(os.path.join(args.source_dir, episode)))
	for data in episode_data:
		if not os.path.isdir(os.path.join(args.source_dir, episode, data)):
			shutil.copy2(os.path.join(args.source_dir, episode, data), os.path.join(args.target_dir, episode, data))
		else:
			shutil.copytree(os.path.join(args.source_dir, episode, data), os.path.join(args.target_dir, episode, data))


def main():
	if not os.path.isdir(args.target_dir):
		os.mkdir(args.target_dir)

	episode_list = sorted(os.listdir(args.source_dir))

	jobs = []
	count = 0
	st = time.time()
	for episode in episode_list:
		count += 1
		if count - 1 < args.start_num or count - 1  > args.end_num:
			continue
		if not os.path.isdir(os.path.join(args.source_dir, episode)):
			shutil.copy2(os.path.join(args.source_dir, episode), os.path.join(args.target_dir, episode))
		else:
			process = multiprocessing.Process(target=copy_episode, args=(episode, ))
			jobs.append(process)
			process.start()

	for process in jobs:
		process.join()

	print ('total time taken: ', time.time()-st)


if __name__ == '__main__':
	global args

	parser = argparse.ArgumentParser()

	parser.add_argument('--source_dir', type=str, help='source directory where data exists')
	parser.add_argument('--target_dir', type=str, help='target directory where to copy the data')
	parser.add_argument('--start_num', type=int, help='cardinality of the start episode (0-indexing)')
	parser.add_argument('--end_num', type=int, help='cardinality of the end episode (0-indexing)')

	args = parser.parse_args()

	main()