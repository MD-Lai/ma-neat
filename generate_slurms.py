# tst = int(tst)
# ban = int(ban)
# cpu = int(cpu)
# sta = int(sta)

test = 6
band = 4
cpu = 32
sta = 0

for tst in range(test):
    for ban in range(band):
        file_name = f"ma_neat_{tst}_{ban}_{cpu}_{sta}-{sta+cpu-1}.slurm"
        contents = f'#!/bin/bash\n\
# Partition for the job:\n\
#SBATCH --partition=physical\n\
\n\
# The name of the job:\n\
#SBATCH --job-name="{tst}_{ban}_{cpu}_{sta}-{sta+cpu-1}"\n\
#SBATCH --output="out_{tst}_{ban}_{cpu}_{sta}-{sta+cpu-1}.out"\n\
\n\
# Maximum number of tasks/CPU cores used by the job:\n\
#SBATCH --ntasks=1\n\
#SBATCH --cpus-per-task={cpu}\n\
\n\
# Send yourself an email when the job:\n\
# aborts abnormally (fails)\n\
#SBATCH --mail-type=FAIL\n\
# begins\n\
#SBATCH --mail-type=BEGIN\n\
# ends successfully\n\
#SBATCH --mail-type=END\n\
\n\
# Use this email address:\n\
#SBATCH --mail-user=derl@student.unimelb.edu.au\n\
\n\
# The maximum running time of the job in days-hours:mins:sec\n\
#SBATCH --time=1-00:00:00\n\
\n\
# Run the job from the directory where it was launched (default)\n\
\n\
module load Python/3.7.1-GCC-6.2.0\n\
python main_mp.py {tst} {ban} {cpu} {sta}\n\
mv results_{tst}_* /data/cephfs/punim1244/maneat\n'

        with open(file_name, 'w', newline='\n') as slurm:
            slurm.write(contents)
        print(f"sbatch {file_name}")