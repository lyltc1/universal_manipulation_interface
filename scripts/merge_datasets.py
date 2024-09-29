''' usage: python merge_datasets.py --sessions data/dataset/fold_pink_towel_hjc_best data/dataset/fold_pink_towel_lyl_right_v3 --output_dir data/dataset/fold_pink_towel_merged_v1'''
import sys
import os

cur_file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(cur_file_path))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import argparse
import zarr
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', nargs='+', help="source folders which contains dataset.zarr.zip")
    parser.add_argument('--output_dir', help="where to save the merged file")
    args = parser.parse_args()

    sessions = args.sessions
    args.output_dir = os.path.join(ROOT_DIR, args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    session = sessions[0]
    dataset_path = os.path.join(ROOT_DIR, session, 'dataset.zarr.zip')
    with zarr.ZipStore(str(dataset_path), mode='r') as zip_store:
        out_replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )
    compressors = out_replay_buffer.get_compressors()
    for session in tqdm(sessions[1:]):
        dataset_path = os.path.join(ROOT_DIR, session, 'dataset.zarr.zip')
        with zarr.ZipStore(str(dataset_path), mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )
        tmp_replay_buffer = ReplayBuffer.create_empty_zarr()

        while replay_buffer.n_episodes:
            if replay_buffer.n_episodes % 2 == 0:
                print(f"for this session, {replay_buffer.n_episodes} left")
            episode = replay_buffer.pop_episode()
            tmp_replay_buffer.add_episode(episode, compressors=compressors)
        while tmp_replay_buffer.n_episodes:
            if tmp_replay_buffer.n_episodes % 2 == 0:
                print(f"for this session, {tmp_replay_buffer.n_episodes} left")
            episode = tmp_replay_buffer.pop_episode()
            out_replay_buffer.add_episode(episode, compressors=compressors)


    # dump to disk
    output_file = os.path.join(ROOT_DIR, args.output_dir, 'dataset.zarr.zip')
    print(f"Saving ReplayBuffer to {output_file}")
    with zarr.ZipStore(output_file, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {out_replay_buffer.n_episodes} videos used in total!")


if __name__ == "__main__":
    main()