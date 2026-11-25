"""Guard for the new-day confirmation: does today's preprocessing reproduce the graphs the models were trained on?

The original run kept edges.csv, nodes.csv and graph_stats.csv for every day (the graph pickle was not kept for all
days), so the check compares those three files byte for byte. They are written by the same code as the pickle, from the
same per-window aggregates. Run from C:\\Users\\rvais\\twin:
  python check_reprocess.py --new D:\\unsw\\check_out\\graphs_300s\\16-10-09 --old outputs\\graphs_300s\\16-10-09
"""
import argparse, hashlib, sys
from pathlib import Path

FILES = ['edges.csv', 'nodes.csv', 'graph_stats.csv']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    bad = []
    for name in FILES:
        a, b = Path(args.new) / name, Path(args.old) / name
        if not a.exists() or not b.exists():
            bad.append(f'{name}: missing ({"new" if not a.exists() else "old"})'); continue
        same = sha(a) == sha(b)
        print(f'{name:16s} {"identical" if same else "DIFFERENT"}  ({b.stat().st_size} bytes)')
        if not same: bad.append(name)
    if bad:
        print('CHECK FAILED: the preprocessing does not reproduce the existing graphs:', ', '.join(bad)); sys.exit(1)
    print('CHECK PASSED: the preprocessing reproduces the existing graphs exactly.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--new', required=True); p.add_argument('--old', required=True)
    main(p.parse_args())
