#!/usr/bin/env python3
import json, pathlib, simc, time

PROFILE = "windpaws20241225.simc"
PRELUDE = []
# PRELUDE = ["input=repro.simc"]
OUT_PATH = "/dev/shm/simc_single_output.json"

last_time = time.time()
j = simc.run([*PRELUDE, "input=" + PROFILE])
print('time =', time.time() - last_time, 's')

for k, v in simc.buffed_stats(j).items():
    print('stat.' + k, '=', v)

dps = j['sim']['players'][0]['collected_data']['dps']
print('mean_dps =', dps['mean'], "±", dps['mean_std_dev'],
      f"({dps['mean_std_dev'] / dps['mean'] * 100:.1g}%)")

pathlib.Path(OUT_PATH).write_text(json.dumps(j, indent=4))
