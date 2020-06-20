import sys
_, tsts, bdts, saveas = sys.argv

tsts = [int(tst) for tst in tsts.split(",")]
bdts = [int(bdt) for bdt in bdts.split(",")]

with open(saveas,'w') as out:
    cmd = ""
    for tst in tsts:
        for bdt in bdts:
            # cmd += f"\
            cmd += f"python3 main_mp2.py {tst} {bdt} 8 0\n\
python3 main_mp2.py {tst} {bdt} 8 8\n\
python3 main_mp2.py {tst} {bdt} 8 16\n\
python3 main_mp2.py {tst} {bdt} 8 24\n\
if not exist spartan\\test_{tst}_{bdt} mkdir spartan\\test_{tst}_{bdt}\n\
MOVE results_{tst}_{bdt}_* spartan\\test_{tst}_{bdt}\\\n"

    out.write(cmd)