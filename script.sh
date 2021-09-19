#!/bin/sh
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=40
#SBATCH --time=3-0:00:00
#SBATCH --job-name="NASDAQHigher"
#SBATCH --mail-user=lwsada002@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

source venv/bin/activate
python exp.py NASDAQ