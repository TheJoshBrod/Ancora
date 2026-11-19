import argparse
import io
import logging
import sys

from ancora.data.audio.services import ENCODERS
from ancora.data.preprocess import run


def main():
    parser = argparse.ArgumentParser(
        description='Clone Hero song folders and .sng files -> catalog, manifest, tokens and '
        'audio features. See docs/preprocessing.md for outputs.'
    )
    parser.add_argument(
        '--source', required=True, nargs='+', help='Folders of song folders and/or .sng files'
    )
    parser.add_argument('--out', default='data', help='Output folder (default: ./data)')
    parser.add_argument('--catalog-only', action='store_true', help='Write catalog.csv and stop')
    parser.add_argument('--limit', type=int, default=None, help='Process at most N new songs')
    parser.add_argument(
        '--features',
        nargs='+',
        default=['encodec'],
        choices=list(ENCODERS),
        help='Audio features to save (default: encodec)',
    )
    parser.add_argument(
        '--guitar-stem',
        action='store_true',
        help='Also compute features from a separate guitar stem when the song has one',
    )
    parser.add_argument(
        '--primary-only',
        action='store_true',
        help='Process only the primary chart of each song, not every copy',
    )
    parser.add_argument('--reprocess', action='store_true', help='Re-run songs already marked done')
    args = parser.parse_args()

    if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    run(
        args.source,
        args.out,
        catalog_only=args.catalog_only,
        limit=args.limit,
        features=args.features,
        guitar_stem=args.guitar_stem,
        primary_only=args.primary_only,
        reprocess=args.reprocess,
    )


if __name__ == '__main__':
    main()
