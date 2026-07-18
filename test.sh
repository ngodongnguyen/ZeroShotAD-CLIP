
device=0

NOTIFY_SCRIPT=./scripts/notify.py

# LOG=${save_dir}"res.log"
# echo ${LOG}
# depth=(9)
# n_ctx=(12)
# t_n_ctx=(4)
# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA dataset
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale
#         save_dir=./checkpoints/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python test.py --dataset mvtec \
#         --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/mvtecdataset --save_path ./results/${base_dir}/mvtec \
#         --checkpoint_path ${save_dir}epoch_15.pth \
#         --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
#         --print_top_k 10
#     wait
#     done
# done
# python ${NOTIFY_SCRIPT} --mode test --dataset mvtec --log_path ./results/${base_dir}/mvtec/log.txt --exp_name ${base_dir}

# LOG=${save_dir}"res.log"
# echo ${LOG}
# depth=(9)
# n_ctx=(12)
# t_n_ctx=(4)
# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA dataset
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
#         save_dir=./checkpoints/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python test.py --dataset visa \
#         --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/visa --save_path ./results/${base_dir}/visa \
#         --checkpoint_path ${save_dir}epoch_15.pth \
#         --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
#         --print_top_k 10
#     wait
#     done
# done
# python ${NOTIFY_SCRIPT} --mode test --dataset visa --log_path ./results/${base_dir}/visa/log.txt --exp_name ${base_dir}

# LOG=${save_dir}"res.log"
# echo ${LOG}
# depth=(9)
# n_ctx=(12)
# t_n_ctx=(4)
# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA dataset
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
#         save_dir=./checkpoints/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python test.py --dataset mpdd \
#         --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/MPDD --save_path ./results/${base_dir}/mpdd \
#         --checkpoint_path ${save_dir}epoch_15.pth \
#         --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
#         --print_top_k 10
#     wait
#     done
# done
# python ${NOTIFY_SCRIPT} --mode test --dataset mpdd --log_path ./results/${base_dir}/mpdd/log.txt --exp_name ${base_dir}

# LOG=${save_dir}"res.log"
# echo ${LOG}
# depth=(9)
# n_ctx=(12)
# t_n_ctx=(4)
# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA dataset
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
#         save_dir=./checkpoints/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python test.py --dataset DAGM_KaggleUpload \
#         --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/DAGM/DAGM_KaggleUpload --save_path ./results/${base_dir}/dagm \
#         --checkpoint_path ${save_dir}epoch_15.pth \
#         --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
#         --print_top_k 10
#     wait
#     done
# done
# python ${NOTIFY_SCRIPT} --mode test --dataset DAGM_KaggleUpload --log_path ./results/${base_dir}/dagm/log.txt --exp_name ${base_dir}

# LOG=${save_dir}"res.log"
# echo ${LOG}
# depth=(9)
# n_ctx=(12)
# t_n_ctx=(4)
# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA dataset
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
#         save_dir=./checkpoints/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python test.py --dataset SDD \
#         --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/SDD --save_path ./results/${base_dir}/sdd \
#         --checkpoint_path ${save_dir}epoch_15.pth \
#         --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
#         --print_top_k 10

#     wait
#     done
# done
# python ${NOTIFY_SCRIPT} --mode test --dataset SDD --log_path ./results/${base_dir}/sdd/log.txt --exp_name ${base_dir}


# LOG=${save_dir}"res.log"
# echo ${LOG}
# depth=(9)
# n_ctx=(12)
# t_n_ctx=(4)
# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA dataset
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
#         save_dir=./checkpoints/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python test.py --dataset DTD \
#         --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/DTD-Synthetic --save_path ./results/${base_dir}/dtd \
#         --checkpoint_path ${save_dir}epoch_15.pth \
#         --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
#         --print_top_k 10
#     wait
#     done
# done
# python ${NOTIFY_SCRIPT} --mode test --dataset dtd-synthetic --log_path ./results/${base_dir}/dtd/log.txt --exp_name ${base_dir}

# LOG=${save_dir}"res.log"
# echo ${LOG}
# depth=(9)
# n_ctx=(12)
# t_n_ctx=(4)
# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA dataset
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
#         save_dir=./checkpoints/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python test.py --dataset btad \
#         --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/btad/BTech_Dataset_transformed --save_path ./results/${base_dir}/btad \
#         --checkpoint_path ${save_dir}epoch_15.pth \
#         --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
#         --print_top_k 10
#     wait
#     done
# done
#python ${NOTIFY_SCRIPT} --mode test --dataset btad --log_path ./results/${base_dir}/btad/log.txt --exp_name ${base_dir}
LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset brain \
        --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/br35 --save_path ./results/${base_dir}/br35 \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics image-level \
        --print_top_k 10
    wait
    done
done
python ${NOTIFY_SCRIPT} --mode test --dataset br35 --log_path ./results/${base_dir}/br35/log.txt --exp_name ${base_dir}

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset brain \
        --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/BrainMRI --save_path ./results/${base_dir}/BrainMRI \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics image-level \
        --print_top_k 10
    wait
    done
done
python ${NOTIFY_SCRIPT} --mode test --dataset brainmri --log_path ./results/${base_dir}/BrainMRI/log.txt --exp_name ${base_dir}

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset colon \
        --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/CVC/CVC-ClinicDB --save_path ./results/${base_dir}/ClinicDB \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics pixel-level \
        --print_top_k 10
    wait
    done
done
python ${NOTIFY_SCRIPT} --mode test --dataset clinicdb --log_path ./results/${base_dir}/ClinicDB/log.txt --exp_name ${base_dir}

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset colon \
        --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/CVC/CVC-ColonDB --save_path ./results/${base_dir}/ColonDB \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics pixel-level \
        --print_top_k 10
    wait
    done
done
python ${NOTIFY_SCRIPT} --mode test --dataset colondb --log_path ./results/${base_dir}/ColonDB/log.txt --exp_name ${base_dir}

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset Chest \
        --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/Covid --save_path ./results/${base_dir}/Covid \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics image-level \
        --print_top_k 10
    wait
    done
done
python ${NOTIFY_SCRIPT} --mode test --dataset covid --log_path ./results/${base_dir}/Covid/log.txt --exp_name ${base_dir}

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset colon \
        --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/EndoTect_2020_Segmentation_Test_Dataset --save_path ./results/${base_dir}/EndoTect \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics pixel-level \
        --print_top_k 10
    wait
    done
done
python ${NOTIFY_SCRIPT} --mode test --dataset endotect --log_path ./results/${base_dir}/EndoTect/log.txt --exp_name ${base_dir}

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset brain \
        --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/HeadCT_anomaly_detection --save_path ./results/${base_dir}/HeadCT \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics image-level \
        --print_top_k 10
    wait
    done
done
python ${NOTIFY_SCRIPT} --mode test --dataset headct --log_path ./results/${base_dir}/HeadCT/log.txt --exp_name ${base_dir}

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset ISBI \
        --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/ISIC --save_path ./results/${base_dir}/ISIC \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics pixel-level \
        --print_top_k 10
    wait
    done
done
python ${NOTIFY_SCRIPT} --mode test --dataset isic --log_path ./results/${base_dir}/ISIC/log.txt --exp_name ${base_dir}

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset colon \
        --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/CVC/Kvasir --save_path ./results/${base_dir}/Kvasir \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics pixel-level \
        --print_top_k 10
    wait
    done
done
python ${NOTIFY_SCRIPT} --mode test --dataset kvasir --log_path ./results/${base_dir}/Kvasir/log.txt --exp_name ${base_dir}

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset thyroid \
        --data_path /home/ai3/NguyenND/ZeroShotAD-CLIP/data/TN3K --save_path ./results/${base_dir}/TN3K \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics pixel-level \
        --print_top_k 10
    wait
    done
done
python ${NOTIFY_SCRIPT} --mode test --dataset tn3k --log_path ./results/${base_dir}/TN3K/log.txt --exp_name ${base_dir}

python ${NOTIFY_SCRIPT} --mode msg --text "✅ All tests done: ${base_dir}"
