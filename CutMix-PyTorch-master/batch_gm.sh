#!/bin/bash
#SBATCH -A research
#SBATCH --job-name=___
#SBATCH --qos=medium
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --output=output_files/seed_BigGAN__e_img_gen_%j.txt       # Output file.
#SBATCH --mail-type=END                # Enable email about job finish 
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail  
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cudnn/7-cuda-10.0
python train.py \
--net_type resnet \
--dataset seeds \
--depth 200 \
--alpha 240 \
--batch_size 64 \
--lr 0.25 \
--expname resnet \
--epochs 100 \
--beta 1.0 \
--cutmix_prob 0.5 \
--no-verbose