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
        '--include-tokens-only',
        action='store_true',
        help='Also split songs processed with --no-audio',
    )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    split.run(args.data, include_tokens_only=args.include_tokens_only, seed=args.seed)


if __name__ == '__main__':
    main()
