import argparse
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='../carla_results/auto_pilot_v3_42/eval_routes_06_12_23_30_25/lidar_360/0000.npy', help='npy point cloud')

def main():
    pcd_npy = np.load(args.file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_npy[:,0:3])
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    main()