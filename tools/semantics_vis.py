import argparse
import glob
import cv2
import os
import numpy
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='../carla_results/auto_pilot_v3_42/eval_routes_06_12_23_30_25/seg_front', help='Folder with semantic images')

classes = {
  0: [0, 0, 0],         # unlabeled
  1: [0, 0, 0],         # building
  2: [0, 0, 0],         # fence
  3: [0, 0, 0],         # other
  4: [0, 255, 0],       # pedestrian
  5: [0, 0, 0],         # pole
  6: [157, 234, 50],    # road line
  7: [128, 64, 128],    # road
  8: [255, 255, 255],   # sidewalk
  9: [0, 0, 0],         # vegetation
  10: [0, 0, 255],      # vehicle
  11: [0, 0, 0],        # wall
  12: [0, 0, 0],        # traffic sign
  13: [0, 0, 0],        # sky
  14: [0, 0, 0],        # ground
  15: [0, 0, 0],        # bridge
  16: [0, 0, 0],        # rail track
  17: [0, 0, 0],        # guard rail
  18: [0, 0, 0],        # traffic light
  19: [0, 0, 0],        # static
  20: [0, 0, 0],        # dynamic
  21: [0, 0, 0],        # water
  22: [0, 0, 0],        # terrain
  23: [255, 0, 0],      # red light
  24: [0, 0, 0],        # yellow light #TODO should be red
  25: [0, 0, 0],        # green light
  26: [157, 234, 50],   # stop sign
  27: [157, 234, 50],   # stop line marking
    
}

def main():
  dirpath, dirnames, filenames = next(os.walk(args.folder))
  for filename in filenames:
      im =  cv2.imread(dirpath + "/" + filename)[:, :, 2]
      result = numpy.zeros((im.shape[0], im.shape[1], 3))
      for key, value in classes.items():
        result[numpy.where(im == key)] = value
      writepath = str(Path(dirpath).parent)
      Path(writepath + "/converted/").mkdir(parents=True, exist_ok=True)
      cv2.imwrite(writepath + "/converted/" + filename, result)

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    main()