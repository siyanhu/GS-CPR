#!/bin/bash
pes=('ace' 'marepo' 'dfnet' 'glace')
scenes=('chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs')

for pe in "${pes[@]}"
do
    for scene in "${scenes[@]}"
    do
        scene_path="7scenes/scene_${scene}/test"
        model_path="7scenes/scene_${scene}/train"
        exp_name='output'

        python render_pred_7s.py -s "data/${scene_path}" -m "data/${model_path}/${exp_name}" \
        --render_scene "${scene}" --pose_estimator "${pe}"
    done
done
