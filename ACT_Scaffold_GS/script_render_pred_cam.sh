pes=('dfnet' 'ace' 'glace')
scenes=('KingsCollege' 'ShopFacade' 'OldHospital' 'StMarysChurch')

for pe in "${pes[@]}"
do
    for scene in "${scenes[@]}"
    do
    scene_path="cambridge/scene_${scene}/test"
    model_path="cambridge/scene_${scene}/train"
    exp_name='output' 
    python render_pred_cam_act.py -s data/${scene_path} -m data/${model_path}/${exp_name} \
        --render_scene "${scene}" --pose_estimator "${pe}"
    done
done