# export OMPI_MCA_btl_cuda=1  
export AZFUSE_USE_FUSE=0 NCCL_ASYNC_ERROR_HANDLING=0  CUDA_VISIBLE_DEVICES=4,5,6,7
echo $CUDA_VISIBLE_DEVICES

python -m torch.distributed.launch --nproc_per_node=4 --master_port=36688 --use_env finetune_sdm_yaml.py   \
    --cf config/disco_w_tm_samenoise/yz_tiktok_S256L16_xformers_tsv_temdisco_temp_attn.py \
    --do_train  \
    --root_dir runtest/ \
    --local_train_batch_size 2 \
    --local_eval_batch_size 2 \
    --log_dir runtest/exp/s3_tem_t_samenoise \
    --epochs 40 --deepspeed \
    --eval_step 1000 --save_step 1000 \
    --gradient_accumulate_steps 2 \
    --learning_rate 1e-4 --fix_dist_seed --loss_target "noise" \
    --train_yaml  /data/fanghaipeng/project/DYL/Resources/TikTok_finetuning/composite_offset/train_TiktokDance-poses-masks.yaml\
    --val_yaml /data/fanghaipeng/project/DYL/Resources/TikTok_finetuning/composite_offset/new10val_TiktokDance-poses-masks.yaml \
    --unet_unfreeze_type "tem" \
    --refer_sdvae \
    --ref_null_caption False \
    --combine_clip_local --combine_use_mask \
    --train_sample_interval 4 \
    --eval_sample_interval 16 \
    --nframe 16 \
    --frame_interval 1 \
    --conds "poses" "masks" \
    --drop_ref 0.05  \
    --guidance_scale 1.5 \
    --pretrained_model /data/fanghaipeng/project/DYL/Resources/checkpoints/tiktok_cfg.pt  \
    --eval_before_train True    \
    --resume