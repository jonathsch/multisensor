# Multi-Sensor Modalities for Vehicle Control

Code for training a multimodal model robust to sensor failure. The model is trained with pseudolidar data as one of the inputs  with inference being done using Lidar data.

## Setup

This implementation uses Python 3.7, CARLA 0.9.10, and PyTorch 1.10 with CUDA 11.3. We recommend using conda to manage the environment.

Set up the environment by running:
```bash
conda create -n pseudolidar python=3.7
conda activate pseudolidar
pip install -r requirements.txt
```

## Demo

`pseudolidar_demo.ipynb` contains a demo visualizing the different versions of pseudo-LiDAR

## Prepare Data

We use a subset of the [Transfuser](https://github.com/autonomousvision/transfuser) dataset for training all of our models. To download the dataset, follow the instructions on the respective GitHub page liked above.

To generate pseudo-LiDAR data, run the following script for every town you want to use:
```bash
python pseudolidar/depthmap2lidar.py --town DATASET_ROOT_PATH/Town{TOWN}_short --extrinsics pseudolidar/extrinsics.json --fov 100 --image_dim 400 300
```

Alternatively, use the `preprocessing.ipynb` notebook.

## Training

In `transfuser/config.py`:

1. If you downloaded the dataset yourself, set `root_dir`  accordingly.

2. If you want to use pseudo-LiDAR for training, set `use_pseudolidar = True`. If not, make sure it is set to `False`

3. List the towns you want to use for training under `train_towns`. If you use pseudo-LiDAR, make sure that you generated the required data for every town.


To train a model with no sensor failure, run:
```bash
python transfuser/train.py --batch_size 8 --use-classification-branch
```

To train a model with random sensor failures, run:
```bash
python transfuser/train.py --batch_size 8 --sensor-failures rgb lidar --use-classification-branch
```

The training of a robust model is shown in the jupyter notebook `training.ipynb`.

## Evaluation

#### Offline Evaluation
For an offline evaluation, use the notebook `test.ipynb`

Note that this script will act according to the settings in `transfuser/config.py`!

#### Online Evaluation
For an online evaluation, open `leaderboard/scripts/run_evaluation.sh` and set:

1. `TEAM_AGENT` to one of the following: 
    - `leaderboard/team_code/transfuser_agent_classification_branch.py` - Runs with no sensor failure
    - `leaderboard/team_code/transfuser_agent_classification_branch_nolidar.py` - Runs without LiDAR
    - `leaderboard/team_code/transfuser_agent_classification_branch_norgb.py` - Runs without RGB

2. `TEAM_CONFIG` to the folder containing the model checkpoint / weights.

3. `CHECKPOINT_ENDPOINT` to the desired results file.

Start a CARLA server by running:
```bash
./start_carla_server.sh
```

Finally, start the evaluation by running:
```bash
CUDA_VISIBLE_DEVICES=0 ./leaderboard/scripts/run_evaluation.sh
```

Note that this may take up to 6 - 8 hours.
