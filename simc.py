import json, os, pathlib, subprocess, sys, tempfile

def recurse_json(p, j):
    if isinstance(j, dict):
        for k, v in j.items():
            yield from recurse_json(p + [k], v)
    elif isinstance(j, list):
        for k, v in enumerate(j):
            yield from recurse_json(p + [k], v)
    else:
        yield p, j

def search_json_key(j, query):
    for k, v in recurse_json([], j):
        if query in repr((k, v)):
            print(k, v)

def run(args):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.json")
        cmd = ["simc", *args, "json2=" + output_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode:
            sys.stderr.buffer.write(proc.stderr)
            sys.stderr.buffer.write(proc.stdout)
            raise SystemExit(proc.returncode)
        return json.loads(pathlib.Path(output_path).read_text())

def rescale_gear(j, target_ilevel, clamp=True):
    ilevels = []
    gear = []
    for k, v in j['sim']['players'][0]['gear'].items():
        ilevel = target_ilevel
        if clamp:
            ilevel = min(v['ilevel'], ilevel)
        ilevels.append(ilevel)
        gear.append(f"{k}={v['encoded_item']},ilevel={ilevel}")
    return gear, sum(ilevels) / len(ilevels)

def buffed_stats(j):
    buffed_stats = j['sim']['players'][0]['collected_data']['buffed_stats']
    stats = {}
    for k, v in buffed_stats['attribute'].items():
        stats[k] = v
    for k, v in buffed_stats['stats'].items():
        if k.endswith('_rating'):
            stats[k.removesuffix('_rating')] = v
    return stats
