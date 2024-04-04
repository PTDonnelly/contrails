"""#!/bin/bash
#SBATCH --job-name=ERA5
#SBATCH --output=/data/pdonnelly/era5/build_era5.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8GB

# Purge all modules to prevent conflict with current environnement
module purge

# Load necessary modules
module load python/meso-3.8

python build_era5_dataset.py
"""