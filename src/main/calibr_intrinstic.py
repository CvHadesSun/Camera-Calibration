import sys
import os
cur = os.getcwd()
sys.path.insert(0,os.path.join(cur,'../lib'))

from intrinstic import calib_intri_share,calib_intri


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../data')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--share_intri', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--remove', action='store_true')
    args = parser.parse_args()
    if args.share_intri:
        calib_intri_share(args.path, step=args.step)
    else:
        calib_intri(args.input, step=args.step)