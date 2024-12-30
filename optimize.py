#!/usr/bin/env python3
import functools, matplotlib, numpy, pathlib, simc, time
from concurrent import futures
from matplotlib import pyplot
from numpy import linalg, random
from scipy import optimize, stats

CMAP = pyplot.get_cmap('viridis')
PROFILE = "windpaws20241225.simc"
STATS = ["crit", "haste", "mastery", "versatility"]
NUM_WORKERS = 20
TARGET_ERROR = 0.5
NUM_PARTICLES = 20

preflight = simc.run(["input=preflight.simc", "input=" + PROFILE])
dps0 = preflight['sim']['players'][0]['collected_data']['dps']['mean']
orig_stats = simc.buffed_stats(preflight)
s0 = numpy.array([orig_stats[stat] for stat in STATS])
st = s0.sum()
x0 = s0 / st

DE_F = (0.5, 1)
DE_CR = 0.7

@functools.lru_cache(maxsize=NUM_PARTICLES * 5)
def calc_dps_inner(err, s):
    # time.sleep(0.1)
    # p = (1+s[0])**0.8 * (1+s[1]) * (1+s[2])**1.2 * (1+s[3])**0.9
    # print(p)
    # return p, p * err

    s = numpy.array(s)
    enchants = [f"enchant_{stat}_rating={d}" for stat, d in zip(STATS, s - s0)]
    try:
        j = simc.run([
            "threads=1",
            f"target_error={err * 100}",
            "input=" + PROFILE,
            *enchants,
        ])
    except Exception as e:
        raise ValueError(repr(enchants)) from e
    dps = j['sim']['players'][0]['collected_data']['dps']
    return dps['mean'], dps['mean_std_dev']

def calc_dps(err, s):
    return calc_dps_inner(err, tuple(s))

rng = random.default_rng()
fig, [ax, ax2, ax3] = pyplot.subplots(3)
ax.set_xticks(range(len(STATS)))
ax.set_xticklabels(STATS)
ax.set_ylim(0, st)

y_history = []
[y_plot] = ax2.plot([0])
ax2.set_ylabel("min(dps)")

y_std_history = []
[y_std_plot] = ax3.plot([0])
ax3.set_ylabel("std(dps)")

def normalize(x):
    x = numpy.transpose(x)
    x = x - x.min(axis=0)
    s = x.sum(axis=0)
    x = x / s
    return numpy.transpose(x)

xs = normalize(rng.random(size=(NUM_PARTICLES, len(STATS))))
plots = []
for x in xs:
    plots.extend(ax.plot(st * x, marker='x'))

def walk(ix):
    i, x = ix
    y, ey = calc_dps(target_error, st * x)
    indices = list(range(len(xs)))
    indices.remove(i)
    rng.shuffle(indices)
    a, b, c = indices[:3]
    mask = rng.random(size=x.shape)
    mask[rng.integers(0, len(x))] = 0
    f = rng.uniform(*DE_F)
    new_x = normalize(numpy.where(mask < DE_CR, xs[a] + f * (xs[b] - xs[c]), x))
    new_y, new_ey = calc_dps(target_error, st * new_x)
    # print(x, y, ey, "-->", new_x, new_y)
    new_r_y = rng.normal(loc=new_y, scale=new_ey)
    r_y = rng.normal(loc=y, scale=ey)
    if new_r_y > r_y:
        return new_x, new_r_y
    else:
        return x, r_y

with futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
    target_error = TARGET_ERROR
    not_decreasing = 0
    while True:
        t = time.time()
        y_min = numpy.inf
        y_max = -numpy.inf
        xs, ys = zip(*ex.map(walk, enumerate(xs)))
        y_min = min(y_min, *ys)
        y_max = max(y_max, *ys)
        y_std = numpy.subtract(*numpy.percentile(ys, [75, 25]))
        print('time_taken =', time.time() - t, ', target_error =', target_error, ', std(ys) =', y_std, not_decreasing)
        if y_std_history and y_std >= y_std_history[-1]:
            not_decreasing += 1
            if not_decreasing >= 3:
                target_error *= 0.5
                not_decreasing = 0
        else:
            not_decreasing = 0
        y_history.append(numpy.min(ys))
        y_history = y_history[-50:]
        y_std_history.append(y_std)
        y_std_history = y_std_history[-50:]

        ax.set_title(f'{y_min} - {y_max}')
        for plot, (x, y) in zip(plots, sorted(zip(xs, ys), key=lambda u: u[1])):
            plot.set_color(CMAP((y - y_min) / max(1e-300, y_max - y_min)))
            plot.set_ydata(st * x)
        y_plot.set_data(range(len(y_history)), y_history)
        ax2.set_xlim(0, len(y_history))
        ax2.set_ylim(numpy.min(y_history), numpy.max(y_history))
        y_std_plot.set_data(range(len(y_std_history)), y_std_history)
        ax3.set_xlim(0, len(y_std_history))
        ax3.set_ylim(0, numpy.max(y_std_history))
        pyplot.pause(0.001)
