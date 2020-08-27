#!/bin/bash
#SBATCH -A research
#SBATCH --job-name=___
#SBATCH --qos=medium
#SBATCH -n 30
#SBATCH --gres=gpu:1
#SBATCH --output=output_log_files/Labls_all_Q%j.out       # Output file.
#SBATCH --mail-type=END                # Enable email about job finish 
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail  
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cudnn/7-cuda-10.0

#source venv/bin/activate
# cd Glow_pyTorch/glow/
 # python3 printing_the_files_name_in_the_directory.py
cd seeds_dataset
# mpiexec -n 3 python download_images.py
python download_images.py 