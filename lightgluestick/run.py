import argparse
import os
from os.path import join

import cv2
import torch
from matplotlib import pyplot as plt
import csv
from .utils import batch_to_np, numpy_image_to_torch
from .viz2d import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from .two_view_pipeline import TwoViewPipeline


def main():
    # Parse input parameters
    parser = argparse.ArgumentParser(
        prog='LightGlueStick Demo',
        description='Demo app to show the point and line matches obtained by LightGlueStick')
    parser.add_argument('-img0', default=('/home/container_user/husky_data/src/data/calib_best/5/lidar_result/intensity_equirectangular.png'))
    parser.add_argument('-imgs', nargs='+', required=True, help="List of 5 images")
    parser.add_argument('--max_pts', type=int, default=1000)
    parser.add_argument('--max_lines', type=int, default=300)
    parser.add_argument('--depth_confidence', type=float, default=-1.0)
    parser.add_argument('--skip-imshow', default=False, action='store_true')
    args = parser.parse_args()

    # Evaluation config
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': False,
        'extractor': {
            "name": "wireframe",
            "point_extractor": {
                "name": "superpoint",
                "trainable": False,
                "dense_outputs": True,
                "max_num_keypoints": 2048,
                "force_num_keypoints": False,
            },
            "line_extractor": {
                "name": "lsd",
                "trainable": False,
                "max_num_lines": 250,
                "force_num_lines": False,
                "min_length": 15,
            },
            "wireframe_params": {
                "merge_points": True,
                "merge_line_endpoints": True,
                "nms_radius": 3,
            },
        },
        'matcher': {
            'name': 'lightgluestick',
            'depth_confidence': args.depth_confidence,
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipeline_model = TwoViewPipeline(conf).to(device).eval()

    gray0 = cv2.imread(args.img0, 0) # This is the LiDAR intensity image    
    torch_0 = numpy_image_to_torch(gray0)
    torch_0  = torch_0.to(device)[None]

    for idx, img_path in enumerate(args.imgs):
        print(f"Processing img0 -> img{img_path}")

        gray_i = cv2.imread(img_path, 0)
        torch_i = numpy_image_to_torch(gray_i)
        torch_i = torch_i.to(device)[None]

        scene = {'view0': {"image": torch_0}, 'view1': {"image": torch_i}}
        
        pred = pipeline_model(scene)
        pred = batch_to_np(pred)
        
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0 = pred["matches0"]

        valid_matches = m0 != -1
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches]
        matched_kps1 = kp1[match_indices]

        csv_out = f"/home/container_user/husky_data/src/data/calib_best/5/lightglue_equi/matches_image0_to_{idx+1}.csv"
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)

        with open(csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x0", "y0", "x1", "y1"])
            for (x0, y0), (x1, y1) in zip(matched_kps0, matched_kps1):
                writer.writerow([float(x1), float(y1), float(x0), float(y0)])
        print(f"Saved: {csv_out}")

        plot_images([gray0, gray_i], ['Image 0', f'Image {idx+1}'], dpi=200, pad=2.0)
        plot_matches(matched_kps0, matched_kps1, lw=1, ps=0, a=0.3)
        if not args.skip_imshow:
            plt.show()

if __name__ == '__main__':
    main()
