"""
Usage:
Testing:
python test_on_training_images.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=data/dataset/scene2_small_various_cup/dataset.zarr.zip
"""
import sys
import os

cur_file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(cur_file_path))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=os.path.join(ROOT_DIR, "diffusion_policy", "config"))
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    output_dir = os.path.join(ROOT_DIR, "data/outputs", cfg.output_dir)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir)
    workspace.val()

if __name__ == "__main__":
    main()
