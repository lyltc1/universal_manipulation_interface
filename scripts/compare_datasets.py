''' usage: python merge_datasets.py --sessions data/dataset/scene1_big_cup data/dataset/scene2_small_various_cup --output_dir data/dataset/merged_dataset'''
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
    parser.add_argument('--sessions', nargs='+', help="source folders which contains dataset.zarr.zip")
    args = parser.parse_args()

    sessions = args.sessions
    with ExifToolHelper() as et:
        session1, session2 = sessions
        dataset1_path = os.path.join(ROOT_DIR, session1, 'dataset.zarr.zip')
        dataset2_path = os.path.join(ROOT_DIR, session2, 'dataset.zarr.zip')
        
        with zarr.ZipStore(str(dataset1_path), mode='r') as zip_store1:
            replay_buffer1 = ReplayBuffer.copy_from_store(
                src_store=zip_store1, 
                store=zarr.MemoryStore()
            )
        with zarr.ZipStore(str(dataset2_path), mode='r') as zip_store2:
            replay_buffer2 = ReplayBuffer.copy_from_store(
                src_store=zip_store2, 
                store=zarr.MemoryStore()
            )
        while replay_buffer1.n_episodes and replay_buffer2.n_episodes:
            print(replay_buffer1.n_episodes, replay_buffer2.n_episodes)
            episode_from_1 = replay_buffer1.pop_episode()
            episode_from_2 = replay_buffer2.pop_episode()
            assert episode_from_1.keys() == episode_from_2.keys()
            for key in episode_from_1.keys():
                assert np.array_equal(episode_from_1[key], episode_from_2[key])

    return True




if __name__ == "__main__":
    main()