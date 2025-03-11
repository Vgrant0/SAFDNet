#!/usr/bin/env bash

# 函数：执行指定的命令，如果执行时间小于30分钟，则重试
run_command() {
    local cmd=$1
    local min_time=$2 # 最小时间，以秒为单位

    while true; do
        local start_time=$(date +%s)

        eval $cmd

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        if [ $duration -ge $min_time ]; then
            echo "命令执行成功，耗时 $(($duration / 60)) 分钟."
            break
        else
            echo "命令执行时间少于指定的 $(($min_time / 60)) 分钟，重试..."
            sleep 5  # 短暂休眠后重试
        fi
    done
}


# 指定GPU编号
export CUDA_VISIBLE_DEVICES=0,1

# 使用 run_command 函数执行每个实验s
# 参数：完整的命令字符串和最小执行时间（秒）
train_time=$((10 * 60))  # 10分钟
test_time=$((10))  # 20秒

# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/pxclcd.py 2 --work-dir work_dirs_3090/pxclcd" $train_time

# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/levir.py 2 --work-dir work_dirs_3090/levir" $train_time

# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/sysu.py 2 --work-dir work_dirs_3090/sysu" $train_time

# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/clcd.py 2 --work-dir work_dirs_3090/clcd" $train_time
# run_command "bash tools/smfu_dist_train.sh /home/dongsj/fusiming/mmrscd-master/configs/0EfficientCD/whucd.py 2 --work-dir work_dirs_3090/whucd_dense_2_6_ssim" $train_time

#从检查点恢复
# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/pxclcd.py 2 --work-dir work_dirs_3090_smfu/pxclcd_triangle_dense_2_6_ssim" $train_time
# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/pxclcd.py 2 --work-dir work_dirs_3090_smfu/pxclcd_triangle_dense_2_6_ssim_resume --resume --cfg-options load_from=/home/dongsj/fsm_code/mmrscd/work_dirs_3090_smfu/pxclcd_triangle_dense_2_6_ssim_0.93284506/iter_60000_best.pth" $train_time
# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/sysu.py 2 --work-dir work_dirs_3090_smfu/sysu_triangle_dense_2_6_ssim_resume --resume --cfg-options load_from=/home/dongsj/fsm_code/mmrscd/work_dirs_3090_smfu/sysu_triangle_dense_2_6_ssim_0.70822251/best_mIoU_iter_29000.pth" $train_time
# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/clcd.py 2 --work-dir work_dirs_3090_smfu/clcd_triangle_dense_2_6_ssim_resume --resume --cfg-options load_from=/home/dongsj/fsm_code/mmrscd/work_dirs_3090_smfu/clcd_triangle_dense_2_6_ssim_0.64434123/best_mIoU_iter_8000.pth" $train_time

# clcd
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/clcd.py 2 --work-dir work_ablation/test_clcd
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/clcd.py 2 --work-dir work_ablation/style_selfdiff_levir_0112_resume --resume --cfg-options load_from=/home/jic2/dsj_code/daily_code/25/1/A_New_Start/work_ablation/style_selfdiff_levir_0112_86.05/best_mIoU_iter_40000.pth
# python tools/test.py configs/0EffiicientCD_smfu/clcd.py work_ablation/style_self_mul_diff_clcd/best_mIoU_iter_*.pth --out "work_ablation/style_self_mul_diff_clcd/test_result"
# python tools/general/metric.py --pppred work_ablation/style_self_mul_diff_clcd/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/CLCD/test/label"

# clcd
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/clcd.py 2 --work-dir work_ablation/style_self_mul_diff_clcd
# # bash tools/dist_train.sh configs/0EffiicientCD_smfu/clcd.py 2 --work-dir work_ablation/style_selfdiff_levir_0112_resume --resume --cfg-options load_from=/home/jic2/dsj_code/daily_code/25/1/A_New_Start/work_ablation/style_selfdiff_levir_0112_86.05/best_mIoU_iter_40000.pth
# python tools/test.py configs/0EffiicientCD_smfu/clcd.py work_ablation/style_self_mul_diff_clcd/best_mIoU_iter_*.pth --out "work_ablation/style_self_mul_diff_clcd/test_result"
# python tools/general/metric.py --pppred work_ablation/style_self_mul_diff_clcd/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/CLCD/test/label"

# # sysu
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/sysu.py 2 --work-dir work_ablation/style_self_mul_diff_sysu
# python tools/test.py configs/0EffiicientCD_smfu/sysu.py work_ablation/style_self_mul_diff_sysu/best_mIoU_iter_*.pth --out "work_ablation/style_self_mul_diff_sysu/test_result"
# python tools/general/metric.py --pppred work_ablation/style_self_mul_diff_sysu/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/SYSU-CD/test/label"

# # # pxclcd
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/pxclcd.py 2 --work-dir work_ablation/style_self_mul_diff_pxclcd
# python tools/test.py configs/0EffiicientCD_smfu/pxclcd.py work_ablation/style_self_mul_diff_pxclcd/best_mIoU_iter_*.pth --out "work_ablation/style_self_mul_diff_pxclcd/test_result"
# python tools/general/metric.py --pppred work_ablation/style_self_mul_diff_pxclcd/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/PX-CLCD/test/label"

# # whucd
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/whucd.py 2 --work-dir work_ablation/style_self_mul_diff_whucd
# python tools/test.py configs/0EffiicientCD_smfu/whucd.py work_ablation/style_self_mul_diff_whucd/best_mIoU_iter_*.pth --out "work_ablation/style_self_mul_diff_whucd/test_result"
# python tools/general/metric.py --pppred work_ablation/style_self_mul_diff_whucd/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/WHUCD/label"

# # levir
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/levir.py 2 --work-dir work_ablation/base_style_levir
# # # bash tools/dist_train.sh configs/0EffiicientCD_smfu/levir.py 2 --work-dir work_ablation/style_selfdiff_levir_0112_resume --resume --cfg-options load_from=/home/jic2/dsj_code/daily_code/25/1/A_New_Start/work_ablation/style_selfdiff_levir_0112_86.05/best_mIoU_iter_40000.pth
# python tools/test.py configs/0EffiicientCD_smfu/levir.py work_ablation/base_style_levir/best_mIoU_iter_*.pth --out "work_ablation/base_style_levir/test_result"
# python tools/general/metric.py --pppred work_ablation/base_style_levir/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/LEVIR-CD/test/label"


# # sysu
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/sysu.py 2 --work-dir work_ablation/base_style_sysu
# python tools/test.py configs/0EffiicientCD_smfu/sysu.py work_ablation/base_style_sysu/best_mIoU_iter_*.pth --out "work_ablation/base_style_sysu/test_result"
# python tools/general/metric.py --pppred work_ablation/base_style_sysu/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/SYSU-CD/test/label"

# # pxclcd
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/pxclcd.py 2 --work-dir work_ablation/base_style_pxclcd
# python tools/test.py configs/0EffiicientCD_smfu/pxclcd.py work_ablation/base_style_pxclcd/best_mIoU_iter_*.pth --out "work_ablation/base_style_pxclcd/test_result"
# python tools/general/metric.py --pppred work_ablation/base_style_pxclcd/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/PX-CLCD/test/label"

# # whucd
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/whucd.py 2 --work-dir work_ablation/base_style_whucd
# python tools/test.py configs/0EffiicientCD_smfu/whucd.py work_ablation/base_style_whucd/best_mIoU_iter_*.pth --out "work_ablation/base_style_whucd/test_result"
# python tools/general/metric.py --pppred work_ablation/base_style_whucd/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/WHUCD/label"


# sysu
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/sysu.py 2 --work-dir work_ablation/base_self_sysu
# python tools/test.py configs/0EffiicientCD_smfu/sysu.py work_ablation/base_self_sysu/best_mIoU_iter_*.pth --out "work_ablation/base_self_sysu/test_result"
# python tools/general/metric.py --pppred work_ablation/base_self_sysu/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/SYSU-CD/test/label"

# pxclcd
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/pxclcd.py 2 --work-dir work_ablation/base_self_pxclcd
# python tools/test.py configs/0EffiicientCD_smfu/pxclcd.py work_ablation/base_self_pxclcd/best_mIoU_iter_*.pth --out "work_ablation/base_self_pxclcd/test_result"
# python tools/general/metric.py --pppred work_ablation/base_self_pxclcd/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/PX-CLCD/test/label"

# # whucd
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/whucd.py 2 --work-dir work_ablation/base_self_whucd
# python tools/test.py configs/0EffiicientCD_smfu/whucd.py work_ablation/base_self_whucd/best_mIoU_iter_*.pth --out "work_ablation/base_self_whucd/test_result"
# python tools/general/metric.py --pppred work_ablation/base_self_whucd/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/WHUCD/label"

# # levir
# bash tools/dist_train.sh configs/0EffiicientCD_smfu/levir.py 2 --work-dir work_ablation/base_self_levir
# python tools/test.py configs/0EffiicientCD_smfu/levir.py work_ablation/base_self_levir/best_mIoU_iter_*.pth --out "work_ablation/base_self_levir/test_result"
# python tools/general/metric.py --pppred work_ablation/base_self_levir/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/LEVIR-CD/test/label"

# sysu
bash tools/dist_train.sh configs/0EffiicientCD_smfu/sysu.py 2 --work-dir work_ablation/style_self_mul_diff_sysu
python tools/test.py configs/0EffiicientCD_smfu/sysu.py work_ablation/style_self_mul_diff_sysu/best_mIoU_iter_*.pth --out "work_ablation/style_self_mul_diff_sysu/test_result"
python tools/general/metric.py --pppred work_ablation/style_self_mul_diff_sysu/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/SYSU-CD/test/label"

bash tools/dist_train.sh configs/0EffiicientCD_smfu/sysu.py 2 --work-dir work_ablation/style_self_mul_diff_sysu_1
python tools/test.py configs/0EffiicientCD_smfu/sysu.py work_ablation/style_self_mul_diff_sysu_1/best_mIoU_iter_*.pth --out "work_ablation/style_self_mul_diff_sysu_1/test_result"
python tools/general/metric.py --pppred work_ablation/style_self_mul_diff_sysu_1/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/SYSU-CD/test/label"

bash tools/dist_train.sh configs/0EffiicientCD_smfu/sysu.py 2 --work-dir work_ablation/style_self_mul_diff_sysu_2
python tools/test.py configs/0EffiicientCD_smfu/sysu.py work_ablation/style_self_mul_diff_sysu_2/best_mIoU_iter_*.pth --out "work_ablation/style_self_mul_diff_sysu_2/test_result"
python tools/general/metric.py --pppred work_ablation/style_self_mul_diff_sysu_2/test_result --gggt "/media/jic2/HDD/DSJJ/CDdata/SYSU-CD/test/label"
