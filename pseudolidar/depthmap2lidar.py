import os
from pathlib import Path
from argparse import ArgumentError, ArgumentParser
import json
from typing import List

import numpy as np
from tqdm import tqdm

from lidar_generator import PseudoLidarGenerator
from pcl_generator import PointCloudGenerator

NUM_LIDAR_MEASUREMENTS = 400
MAX_DEPTH_FRONT = 0.05
MAX_DEPTH_SIDE = 0.03

def read_poses(path: os.PathLike) -> List[np.array]:
    with open(path, mode='r') as f:
        poses = json.load(f)
    
    front = PseudoLidarGenerator.load_pose(poses['cam_front'])
    left = PseudoLidarGenerator.load_pose(poses['cam_left'])
    right = PseudoLidarGenerator.load_pose(poses['cam_right'])

    return [front, left, right]

def main():
    ap = ArgumentParser()
    ap.add_argument('--town', type=str, required=True, dest='town_path')
    ap.add_argument('--extrinsics', type=str, required=True, dest='extrinsics_file')
    ap.add_argument('--fov', type=int, required=True, dest='fov')
    ap.add_argument('--image_dim', nargs='+', type=int, required=True, dest='image_dim', help='Image dimensions in x and y direction')
    args = ap.parse_args()

    poses = read_poses(args.extrinsics_file)
    if len(args.image_dim) != 2:
        ap.error(f'Expected 2 parameters for image dimensions, but was {len(args.image_dim)}.')

    pc_generator = PointCloudGenerator(args.fov, (args.image_dim[1], args.image_dim[0]))
    lidar_generator = PseudoLidarGenerator(poses)

    root_path = Path(args.town_path)
    routes = [p for p in root_path.iterdir() if p.is_dir() and p.name.startswith('routes_')]
    for route in tqdm(routes):
        fronts = sorted([p for p in route.joinpath('depth_front').iterdir() if p.is_file() and p.suffix == '.png'])
        lefts = sorted([p for p in route.joinpath('depth_left').iterdir() if p.is_file() and p.suffix == '.png'])
        rights = sorted([p for p in route.joinpath('depth_right').iterdir() if p.is_file() and p.suffix == '.png'])
        
        out_dir = route / 'pseudolidar'
        out_dir.mkdir(exist_ok=True)

        for front, left, right in zip(fronts, lefts, rights):
            pcs = [pc_generator.generate(front, max_depth=MAX_DEPTH_FRONT),
                   pc_generator.generate(left, max_depth=MAX_DEPTH_SIDE),
                   pc_generator.generate(right, max_depth=MAX_DEPTH_SIDE)
            ]

            pc = lidar_generator.generate(pcs)
            mesh = PseudoLidarGenerator.reconstruct_mesh(pc)
            points = PseudoLidarGenerator.raycast_mesh(mesh, n_measurements=NUM_LIDAR_MEASUREMENTS)
            
            np.save(out_dir / f'{front.stem}.npy', points)

if __name__ == '__main__':
    main()
