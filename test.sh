
device=0

depth=(9)
n_ctx=(12)
t_n_ctx=(4)

## visa
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset visa \
        --data_path ./data/visa --save_path ./results/${base_dir}/zero_shot/visa \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
        python scripts/notify.py --dataset visa \
            --log_path ./results/${base_dir}/zero_shot/visa/log.txt \
            --exp_name "${base_dir}"
    wait
    done
done

## mpdd
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset mpdd \
        --data_path ./data/MPDD --save_path ./results/${base_dir}/zero_shot/mpdd \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
        python scripts/notify.py --dataset mpdd \
            --log_path ./results/${base_dir}/zero_shot/mpdd/log.txt \
            --exp_name "${base_dir}"
    wait
    done
done

## dtd
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset DTD \
        --data_path ./data/DTD-Synthetic --save_path ./results/${base_dir}/zero_shot/dtd \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
        python scripts/notify.py --dataset dtd \
            --log_path ./results/${base_dir}/zero_shot/dtd/log.txt \
            --exp_name "${base_dir}"
    wait
    done
done

## sdd
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset KolektorSDD \
        --data_path ./data/KolektorSDD-boxes --save_path ./results/${base_dir}/zero_shot/sdd \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
        python scripts/notify.py --dataset sdd \
            --log_path ./results/${base_dir}/zero_shot/sdd/log.txt \
            --exp_name "${base_dir}"
    wait
    done
done

## dagm
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset DAGM_KaggleUpload \
        --data_path ./data/archive/DAGM_KaggleUpload --save_path ./results/${base_dir}/zero_shot/dagm \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
        python scripts/notify.py --dataset dagm \
            --log_path ./results/${base_dir}/zero_shot/dagm/log.txt \
            --exp_name "${base_dir}"
    wait
    done
done
