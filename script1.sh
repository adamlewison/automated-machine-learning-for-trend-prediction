#!/bin/sh
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=40
#SBATCH --time=5-2:00:00
#SBATCH --job-name="DE_STX40"
#SBATCH --mail-user=lwsada002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

source venv/bin/activate
python exp1.py STX40