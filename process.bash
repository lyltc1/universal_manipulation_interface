FILENAME=small_fold_pink_1
python run_slam_pipeline.py universal_manipulation_interface/data/dataset/$FILENAME &&
python scripts_slam_pipeline/07_generate_replay_buffer.py -o data/dataset/$FILENAME/dataset.zarr.zip data/dataset/$FILENAME

FILENAME=small_fold_pink_2
python run_slam_pipeline.py universal_manipulation_interface/data/dataset/$FILENAME &&
python scripts_slam_pipeline/07_generate_replay_buffer.py -o data/dataset/$FILENAME/dataset.zarr.zip data/dataset/$FILENAME

FILENAME=small_fold_pink_3
python run_slam_pipeline.py universal_manipulation_interface/data/dataset/$FILENAME &&
python scripts_slam_pipeline/07_generate_replay_buffer.py -o data/dataset/$FILENAME/dataset.zarr.zip data/dataset/$FILENAME

FILENAME=small_fold_pink_4
python run_slam_pipeline.py universal_manipulation_interface/data/dataset/$FILENAME &&
python scripts_slam_pipeline/07_generate_replay_buffer.py -o data/dataset/$FILENAME/dataset.zarr.zip data/dataset/$FILENAME

FILENAME=small_fold_pink_5
python run_slam_pipeline.py universal_manipulation_interface/data/dataset/$FILENAME &&
python scripts_slam_pipeline/07_generate_replay_buffer.py -o data/dataset/$FILENAME/dataset.zarr.zip data/dataset/$FILENAME