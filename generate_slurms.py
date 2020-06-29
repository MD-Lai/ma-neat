# tst = int(tst)
# ban = int(ban)
# cpu = int(cpu)
# sta = int(sta)


# tests = [
#     env_Pendulum_v0,
#     env_BipedalWalker_v2,
#     env_BipedalWalkerHardcore_v2,
#     env_LunarLanderContinuous_v2,
#     cls_wine,
#     cls_banknote,
#     cls_MNIST
# ]

run_times = [
    "0-00:30:00",
    "0-06:00:00",
    "0-06:00:00",
    "1-12:00:00",
    "0-02:00:00",
    "0-02:00:00",
    "2-00:00:00",
    "0-01:00:00"
]

tests = [7,4,0,5,1]
bandits = range(17)
cpu = 32
sta = 0

print("#!/bin/bash")
for tst in tests:
    for ban in bandits:
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
#SBATCH --time={run_times[int(tst)]}\n\
\n\
# Run the job from the directory where it was launched (default)\n\
\n\
module load Python/3.7.1-GCC-6.2.0\n\
python main_mp2.py {tst} {ban} {cpu} {sta}\n\
mkdir /data/cephfs/punim1244/maneat/test_{tst}_{ban}\n\
mv results_{tst}_{ban}* /data/cephfs/punim1244/maneat/test_{tst}_{ban}\n'

        with open(file_name, 'w', newline='\n') as slurm:
            slurm.write(contents)
        print(f"sbatch {file_name}")