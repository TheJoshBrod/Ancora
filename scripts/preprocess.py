import argparse
import io
import logging
import sys

from ancora.data.preprocess import run


def main():
    parser = argparse.ArgumentParser(
        description='Clone Hero songs -> EnCodec audio codes + 100ms note tokens. '
        'See docs/preprocessing.md for outputs.'
    )
    parser.add_argument('--source', required=True, help='Folder of song folders and/or .sng files')
    parser.add_argument('--out', default='data', help='Output folder (default: ./data)')
    parser.add_argument('--limit', type=int, default=None, help='Process at most N new songs')
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Skip EnCodec audio encoding (charts/tokens only)',
    )
    parser.add_argument(
        '--reprocess', action='store_true', help='Re-run songs already marked done in the manifest'
    )
    args = parser.parse_args()

    # Fix print encoding on Windows
    if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    run(args.source, args.out, limit=args.limit, no_audio=args.no_audio, reprocess=args.reprocess)


if __name__ == '__main__':
    main()
