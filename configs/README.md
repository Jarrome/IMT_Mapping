## How to configure

For example, ```./config/ifr-fusion-lr-kt0.yaml``` contains several settings:

```
dataset_type: "TUM" # for ICL-NUIM 's TUM format
scene: "0n" # scene 0 with noise
use_gt: False # not use groundtruth trajectory
pose_folder: "./treasure/orbslam2_record/lrkt0n/" # the predicted pose stream from orbslam2
outdir: "./res/lrkt0n_ours/" # where we stores the intermediary mesh
calib: [481.2, 480.0, 319.50, 239.50, 5000.0] # calibration of images

sequence_kwargs: # this is called with within sequence namespace
  path: "./data/ICL_NUIM/lr_kt0n/"
  start_frame: 0
  end_frame: -1                                       # Run all frames
  first_tq: [-1.4, 1.5, 1.5, 0.0, -1.0, 0.0, 0.0]     # Starting pose

# Network parameters
training_hypers: "./treasure/hyper.json"
using_epoch: 600

# Enable visualization
vis: True
resolution: 4

# meshing
max_n_triangles: 4e6
max_std: 0.15 # 0.06

# These two define the range of depth observations to be cropped. Unit is meter.
depth_cut_min: 0.5
depth_cut_max: 5.0

# not exactly used in code, please follow real implementation
meshing_interval: 20
integrate_interval: 20

# Mapping parameters
mapping:
  # Bound of the scene to be reconstructed
  bound_min: [-3.5, -0.5, -2.5]
  bound_max: [4.5, 3.5, 5.5]
  voxel_size: 0.1
  # Prune observations if detected as noise.
  prune_min_vox_obs: 16
  ignore_count_th: 16.0
  encoder_count_th: 600.0
```
