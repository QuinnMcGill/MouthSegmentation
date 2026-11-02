"""Print a table of blendshape values for one frame across several blendshape CSVs.

Usage:
    python compare_blendshapes.py FRAME csv1 [csv2 ...] [--filter mouth]
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frame", type=int, help="frame number to display")
    parser.add_argument("csvs", nargs="+", help="blendshape CSV files")
    parser.add_argument("--filter", default=None, help="only show blendshapes containing this substring (case-insensitive)")
    args = parser.parse_args()

    columns = {}
    for path in args.csvs:
        df = pd.read_csv(path, index_col=0)
        row = df[df["frame"] == args.frame]
        if row.empty:
            print(f"warning: frame {args.frame} not found in {path}")
            continue
        columns[os.path.splitext(os.path.basename(path))[0]] = row.iloc[0].drop("frame")

    if not columns:
        return

    table = pd.DataFrame(columns)
    if args.filter:
        table = table[table.index.str.contains(args.filter, case=False)]
    if len(columns) == 2:
        a, b = table.columns
        table["diff"] = table[b] - table[a]

    print(f"Frame {args.frame}")
    print(table.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
