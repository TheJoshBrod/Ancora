import argparse
import logging
import sys

from ancora.data import split


def main():
    parser = argparse.ArgumentParser(
        description='Song-grouped train/val/test split over manifest.csv. '
        'See docs/preprocessing.md.'
    )
    parser.add_argument(
        '--data', default='data', help='Folder with manifest.csv (output of scripts/preprocess.py)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Only check the existing split for leaks; exit 1 if any',
    )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    sys.exit(split.run(args.data, check=args.check, seed=args.seed))


if __name__ == '__main__':
    main()
