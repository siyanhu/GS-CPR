
#!/bin/bash
#scenes=('chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs')
pes=('ace' 'marepo' 'glace')
#scenes=('apt1_kitchen' 'apt1_living' 'apt2_bed' 'apt2_kitchen' 'apt2_living' 'apt2_luke' 'office1_gates362' 'office1_gates381' 'office1_lounge' 'office1_manolis' 'office2_5a' 'office2_5b')
scenes=('apt1_kitchen')
pe="${pes[0]}"
for pe in "${pes[@]}"
do
    for scene in "${scenes[@]}"
    do
        scene_path="12scenes/scene_${scene}/test"
        model_path="12scenes/scene_${scene}/train"
        exp_name='output'

        python render_pred_12s.py -s "data/${scene_path}" -m "data/${model_path}/${exp_name}" \
        --render_scene "${scene}" --pose_estimator "${pe}"
    done
done