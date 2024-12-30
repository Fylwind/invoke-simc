#!/usr/bin/env python3
import json, os, simc

PROFILE = "windpaws20241225.simc"
# PROFILE = os.path.expanduser("~/stuff/simc/profiles/TWW1/TWW1_Monk_Windwalker.simc")
# PROFILE = os.path.expanduser("~/stuff/simc/profiles/TWW1/TWW1_Rogue_Assassination.simc")

# scale_to_itemlevel doesn't seem to scale stats properly, so we do the scaling manually instead.
preflight = simc.run(["input=preflight.simc", "input=" + PROFILE])
records = {}
for target_ilevel in range(580, 680, 5):
    gear, avg_ilevel = simc.rescale_gear(preflight, target_ilevel=target_ilevel, clamp=False)
    j = simc.run(["target_error=0.2", f"input={PROFILE}", *gear])
    print(json.dumps({
        'avg_ilevel': avg_ilevel,
        'mean_dps': j['sim']['players'][0]['collected_data']['dps']['mean'],
        **simc.buffed_stats(j),
    }))
