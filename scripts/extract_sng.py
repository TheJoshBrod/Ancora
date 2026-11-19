import sys

from ancora.data.sng import extract_sng


def main():
    if len(sys.argv) < 3:
        print('Usage: extract_sng.py <file.sng> <dest_dir>')
        sys.exit(1)
    meta = extract_sng(sys.argv[1], sys.argv[2])
    print('Metadata:')
    for k, v in meta.items():
        print(f'  {k} = {v}')
    print(f'Files extracted to: {sys.argv[2]}')


if __name__ == '__main__':
    main()
