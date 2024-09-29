import sys
import os

cur_file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(cur_file_path))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import argparse
import zarr
import numpy as np
from tqdm import tqdm
from exiftool import ExifToolHelper

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', nargs='+', help="source folders which contains dataset.zarr.zip", default = ['data/dataset/scene1_big_cup', 'data/dataset/scene1_big_cup_reverse_reverse'])
    args = parser.parse_args()

    sessions = args.sessions

    with ExifToolHelper() as et:
        dataset_path1 = os.path.join(ROOT_DIR, sessions[0], 'dataset.zarr.zip')
        dataset_path2 = os.path.join(ROOT_DIR, sessions[1], 'dataset.zarr.zip')

        replay_buffer1 = None
        with zarr.ZipStore(str(dataset_path1), mode='r') as zip_store1:
            replay_buffer1 = ReplayBuffer.copy_from_store(
                src_store=zip_store1, 
                store=zarr.MemoryStore()
            )
        replay_buffer2 = None
        with zarr.ZipStore(str(dataset_path2), mode='r') as zip_store2:
            replay_buffer2 = ReplayBuffer.copy_from_store(
                src_store=zip_store2, 
                store=zarr.MemoryStore()
            )
        while True:
            episode1 = replay_buffer1.pop_episode()
            episode2 = replay_buffer2.pop_episode()
            for k in episode1.keys():
                assert np.all(episode1[k] == episode2[k])
            print(1)

if __name__ == "__main__":
    main()