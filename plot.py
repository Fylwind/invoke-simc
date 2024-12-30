#!/usr/bin/env python3
import pandas, sys
from matplotlib import pyplot

def label(label, d):
    d['label'] = label
    return d

d = pandas.concat([
    label('monkwindpaw', pandas.read_json('data/monkwindpaw2.json', lines=True)),
    label('monkww', pandas.read_json('data/monkww2.json', lines=True)),
    label('rogueas', pandas.read_json('data/rogueas2.json', lines=True)),
])
d = d.pivot(index='avg_ilevel', columns='label', values='mean_dps')
fig, ax = pyplot.subplots()
d.plot(ax=ax, marker='x')
ax.set_yscale('log')
pyplot.show()
