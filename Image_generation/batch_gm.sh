#!/bin/bash
#SBATCH -A research
#SBATCH --job-name=___
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --output=output_files/seed_BigGAN__e_img_gen_%j.txt       # Output file.
#SBATCH --mail-type=END                # Enable email about job finish 
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail  
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cudnn/7-cuda-10.0
python -W ignore train.py \
--experiment_name seeds \
--shuffle --batch_size 256 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 200 \
--num_D_steps 1 --G_lr 2e-4 --D_lr 2e-4 \
--dataset TI64 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--save_every 2000 --num_best_copies 1 --num_save_copies 1 --seed 0 \
--weights_root='weights'  \
--logs_root='logs'  \
--samples_root='samples' \
--test_every -1 \
--historical_save_every 180000 \
--model=BigGAN \
# --load_weights='copy0' \
# --mh_loss \
# --mh_loss_weight 0.05 \
# --resume \
# --load_weights='copy0' \

# python download_images.py 
# --data_root data \
# python main.py 



## For generating samples #### Batch size should be a perfect square
############
############



module load cudnn/7-cuda-10.0
python -W ignore sample.py \
--experiment_name seedsmh \
--shuffle --batch_size 16 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 300 \
--num_D_steps 2 --G_lr 2e-4 --D_lr 2e-4 \
--dataset TI64 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--save_every 2000 --num_best_copies 1 --num_save_copies 1 --seed 0 \
--weights_root='weights'  \
--logs_root='logs'  \
--samples_root='samples' \
--test_every -1 \
--G_eval_mode \
--sample_multiple \
--load_weights 'copy0' \
--model BigGANmh \
--sample_sheets \
--sample_interps \
--sample_classwise \
--sample_sheet_folder_num 124000 \
--mh_loss \
--mh_loss_weight 0.05 \

# --sample_np_mem \

# --sample_classwise \

# --shuffle --batch_size 2500 --G_batch_size 256 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --dataset C100 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 5000 --save_every 2000 --num_best_copies 1 --num_save_copies 1 --seed 0 \
# --data_root /fs/vulcan-scratch/ilyak/locDoc/data \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --experiment_name cifar100_baseline_redo \
# --get_generator_error \
# --dataset_is_fid /scratch0/ilya/locDoc/data/cifar10/fid_stats_cifar10_train.npz \
# --G_eval_mode \
# --sample_multiple \
# --load_weights '100000,090000,080000,070000,060000,050000,040000,030000,020000,010000'



