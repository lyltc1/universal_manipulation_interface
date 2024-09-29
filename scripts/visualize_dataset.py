import sys
import os

cur_file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(cur_file_path))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import random
import argparse
import zarr
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

def visualize_robot0_eef_pos(robot0_eef_pos, robot0_gripper_width):
    """
    可视化机器人末端执行器位置和夹爪宽度。

    参数:
    robot0_eef_pos (numpy.ndarray): 形状为 [N, 3] 的数组，表示末端执行器的位置。
    robot0_gripper_width (numpy.ndarray): 形状为 [N, 1] 的数组，表示夹爪的宽度。
    """
    x = robot0_eef_pos[:, 0]
    y = robot0_eef_pos[:, 1]
    z = robot0_eef_pos[:, 2]
    colors = ['b' if width < 0.075 else 'r' for width in robot0_gripper_width]

    fig = plt.figure(figsize=(15, 6))

    # 3D 位置图
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(x[1:], y[1:], z[1:], c=colors[1:], label='robot0_eef_pos')
    ax1.scatter(x[0], y[0], z[0], marker='*', s = 300, color='g', label='start')
    ax1.plot(x, y, z)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Robot0 End Effector Position')
    ax1.legend()


    # 夹爪宽度图
    ax2 = fig.add_subplot(122)
    scatter = ax2.scatter(range(len(robot0_gripper_width)), robot0_gripper_width, c=colors, label='Gripper Width')
    ax2.plot(robot0_gripper_width)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Gripper Width')
    ax2.set_title('Robot0 Gripper Width (red for close)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="source folders which contains dataset.zarr.zip", default="data/dataset/fold_pink_towel_lyl_right_v2")
    args = parser.parse_args()

    dataset = args.dataset
    dataset_path = os.path.join(ROOT_DIR, dataset, 'dataset.zarr.zip')
    with zarr.ZipStore(str(dataset_path), mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )
    print(f"There are {replay_buffer.n_episodes} in the dataset")
    while True:
        # input("press Enter to continue")
        index = random.randint(0, replay_buffer.n_episodes - 1)
        result = replay_buffer.get_episode(index)
        robot0_eef_pos = result['robot0_eef_pos']  # [N, 3]
        # robot0_eef_rot_axis_angle = result['robot0_eef_rot_axis_angle'] # [N, 3]
        robot0_gripper_width = result['robot0_gripper_width']  # [N, 1]
        visualize_robot0_eef_pos(robot0_eef_pos, robot0_gripper_width)

if __name__ == "__main__":
    main()