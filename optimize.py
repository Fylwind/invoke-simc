#!/usr/bin/env python3
"""Optimizes secondary stats, constrained to the total of all secondary stats.

Example:

    ./optimize.py [--checkpoint=checkpoint.txt] \
        armory=us,<hyphenated-realm>,<character>

Or use "input=windpaws20241225.simc" to pass in a local SimC config.

"""
import argparse, dataclasses, functools, itertools, json, matplotlib, numpy, os, pathlib, simc, tempfile, time
from concurrent import futures
from matplotlib import cm, colors, pyplot
from numpy import linalg, random
from scipy import optimize, stats

def is_decreasing(ys):
    fit, cov = numpy.polyfit(range(len(ys)), ys, deg=1, cov=True)
    return stats.norm.cdf(-fit[0] / cov[0, 0]**0.5)

def test_func(x):
    # Known solution: [0.13501869, 0.24703816, 0.49565162, 0.12229153]
    # => -2.516657625739369
    return (1+s[0])**0.91 * (1+s[1]) * (1+s[2])**1.2 * (1+s[3])**0.9

def normalize(x):
    x = numpy.transpose(x)
    x = x - numpy.clip(-numpy.inf, 0, x.min(axis=0))
    s = x.sum(axis=0)
    x = x / s
    return numpy.transpose(x)

@dataclasses.dataclass
class State:
    target_error: float
    xs: numpy.array

    @staticmethod
    def load(path):
        with open(path) as f:
            j = json.load(f)
        return State(
            target_error=j["target_error"],
            xs=numpy.array(j["xs"]),
        )

    def save(self, path):
        j = {
            "target_error": self.target_error,
            "xs": numpy.array(self.xs).tolist(),
        }
        with tempfile.TemporaryDirectory(os.path.dirname(path)) as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "tmp.txt")
            with open(tmp_path, "w") as f:
                json.dump(j, f)
            os.rename(tmp_path, path)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint")
parser.add_argument("simc_args", nargs="+")
args = parser.parse_args()

STATS = ["crit", "haste", "mastery", "versatility"]
CMAP = pyplot.get_cmap("viridis")
NUM_WORKERS = 20
TARGET_ERROR = 0.5
NUM_PARTICLES = 20
MAX_HISTORY = 200              # affects visualization + nondecreasing checker

# Nondecreasing checker: Review last X steps and check whether it appears to
# be nondecreasing. If so, reduce the target_error.  Higher threshold =
# higher chance of false positive = more pron         e to reduction.
NONDECREASING_STEPS = 5
NONDECREASING_THRESHOLD = 0.20

# Differential Evolution parameters
DE_F = (0.5, 1)
DE_CR = 0.7

preflight = simc.run(["input=preflight.simc", *args.simc_args])
dps0 = preflight["sim"]["players"][0]["collected_data"]["dps"]["mean"]
orig_stats = simc.buffed_stats(preflight)
s0 = numpy.array([orig_stats[stat] for stat in STATS])
st = s0.sum()
x0 = s0 / st

rng = random.default_rng()
if args.checkpoint and os.path.exists(args.checkpoint):
    state = State.load(args.checkpoint)
else:
    state = State(
        xs=normalize(rng.random(size=(NUM_PARTICLES, len(STATS)))),
        target_error=TARGET_ERROR,
    )
num_particles, _ = state.xs.shape

@functools.lru_cache(maxsize=num_particles * 5)
def calc_dps_inner(err, s):
    if False:                           # for testing
        time.sleep(0.1)
        p = test_func(numpy.array(s) / st)
        return p, p * err

    s = numpy.array(s)
    enchants = [f"enchant_{stat}_rating={d}" for stat, d in zip(STATS, s - s0)]
    try:
        j = simc.run([
            "threads=1",
            f"target_error={err * 100}",
            *args.simc_args,
            *enchants,
        ])
    except Exception as e:
        raise ValueError(repr(enchants)) from e
    dps = j["sim"]["players"][0]["collected_data"]["dps"]
    return dps["mean"], dps["mean_std_dev"]

def calc_dps(err, s):
    return calc_dps_inner(err, tuple(s))

fig, [ax, ax3] = pyplot.subplots(2)
colorbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(), cmap=CMAP),
                        orientation="horizontal")
ax.set_xticks(range(len(STATS)))
ax.set_xticklabels(STATS)
ax.set_ylim(0, st)

y_disp_history = []
[y_disp_plot] = ax3.plot([0], color="blue")
ax3.set_ylabel("dispersion(dps) (blue)")

target_error_history = []
ax3b = ax3.twinx()
[target_error_plot] = ax3b.plot([0], color="red")
ax3b.set_ylabel("target_error (red)")

plots = []
for x in state.xs:
    plots.extend(ax.plot(st * x, marker="x"))

def walk(ix):
    i, x = ix
    y, ey = calc_dps(state.target_error, st * x)
    indices = list(range(len(state.xs)))
    indices.remove(i)
    rng.shuffle(indices)
    a, b, c = indices[:3]
    mask = rng.random(size=x.shape)
    mask[rng.integers(0, len(x))] = 0
    f = rng.uniform(*DE_F)
    candidate_x = state.xs[a] + f * (state.xs[b] - state.xs[c])
    new_x = normalize(numpy.where(mask < DE_CR, candidate_x, x))
    new_y, new_ey = calc_dps(state.target_error, st * new_x)
    # print(x, y, ey, "-->", new_x, new_y)
    new_r_y = rng.normal(loc=new_y, scale=new_ey)
    r_y = rng.normal(loc=y, scale=ey)
    if new_r_y > r_y:
        return new_x, new_r_y
    else:
        return x, r_y

with futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
    i = 0
    while True:
        t = time.time()
        y_min = numpy.inf
        y_max = -numpy.inf
        state.xs, ys = zip(*ex.map(walk, enumerate(state.xs)))
        y_min = min(y_min, *ys)
        y_max = max(y_max, *ys)
        y_disp = numpy.subtract(*numpy.percentile(ys, [75, 25]))
        num_at_current_err = len(list(itertools.takewhile(
            lambda e: e == state.target_error,
            reversed(target_error_history),
        )))
        best = numpy.argmax(ys)

        if args.checkpoint:
            state.save(args.checkpoint)

        i += 1
        y_disp_history.append(y_disp)
        y_disp_history = y_disp_history[-MAX_HISTORY:]
        target_error_history.append(state.target_error)
        target_error_history = target_error_history[-MAX_HISTORY:]
        decreasing = numpy.nan
        if num_at_current_err > 5:
            decreasing = is_decreasing(y_disp_history[-NONDECREASING_STEPS:])
            if decreasing < NONDECREASING_THRESHOLD:
                state.target_error *= 0.5

        print(f"{i}\tdps={numpy.max(ys):.6g}\tduration={time.time() - t:.1f}\ttarget_error={state.target_error:.2g}\tdecreasing={decreasing:.1f}\t{state.xs[best]}")

        norm = colors.Normalize(vmin=y_min, vmax=y_max)
        colorbar.update_normal(cm.ScalarMappable(norm=norm, cmap=CMAP))
        ordered_xys = sorted(zip(state.xs, ys), key=lambda u: u[1])
        for plot, (x, y) in zip(plots, ordered_xys):
            plot.set_color(CMAP(norm(y)))
            plot.set_ydata(st * x)
        ax.set_title(f"{y_min} - {y_max}")
        ax.relim()
        ax.autoscale_view()
        y_disp_plot.set_data(range(len(y_disp_history)), y_disp_history)
        ax3.relim()
        ax3.autoscale_view()
        target_error_plot.set_data(range(len(target_error_history)), target_error_history)
        ax3b.relim()
        ax3b.autoscale_view()
        pyplot.pause(1e-4)
