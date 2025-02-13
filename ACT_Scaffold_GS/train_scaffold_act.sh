#!/bin/bash

# Function to generate a random number in the range [min, max]
function rand() {
    min=$1
    max=$(($2 - $min + 1))
    num=$(date +%s%N)
    echo $(($num % $max + $min))  
}

# Default parameters
iterations=30_000
warmup="False"
lod=0
appearance_dim=0
ratio=1
voxel_size=0.001
update_init_factor=4
exp_name="output" 
gpu=0

# Generate random port
port=$(rand 10000 30000)

# Process arguments passed to the script
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        --lod) lod="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --warmup) warmup="$2"; shift ;;
        --voxel_size) voxel_size="$2"; shift ;;
        --update_init_factor) update_init_factor="$2"; shift ;;
        --appearance_dim) appearance_dim="$2"; shift ;;
        --ratio) ratio="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Timestamp for output directories
time=$(date "+%Y-%m-%d_%H:%M:%S")

scenes=("ShopFacade")
# Main loop for training each scene
for render_scene in "${scenes[@]}"; do
    # Set the scene path
    scene="cambridge/scene_${render_scene}/train"
    data="cambridge/scene_${render_scene}"
    # Call training script
    # add --eval to sample some validation set from training set, see scene/dataset_readers.py
    if [ "$warmup" = "True" ]; then
        python train_act.py -s data/${scene} --lod ${lod} --gpu ${gpu} --voxel_size ${voxel_size} \
        --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup \
        --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time --render_scene ${render_scene}
    else
        python train_act.py -s data/${scene} --lod ${lod} --gpu ${gpu} --voxel_size ${voxel_size} \
        --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} \
        --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time --render_scene ${render_scene}
    fi
done
