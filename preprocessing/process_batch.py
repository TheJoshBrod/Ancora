"""Batch preprocessing: Clone Hero songs -> EnCodec audio codes + 100ms note tokens.

The source directory may contain extracted song folders (notes.chart / notes.mid,
song.ini, audio stems) and/or packed .sng files. .sng files are extracted into
<out>/extracted/<name>/ first, then processed like any other folder.

Outputs (under --out):
  tokens/<song_id>/audio_codes.npy     EnCodec codes (skipped with --no-audio)
  tokens/<song_id>/tokens_100ms.npy    fret-bitmask tokens on a 100ms grid
  tokens/<song_id>/sustain_100ms.npy   sustain length (s) per grid cell
  tokens/<song_id>/events_raw.npz      (time_s, bitmask, duration_s) events
  manifest.csv                         one row per song with QA flags

Usage:
  python process_batch.py --source "/mnt/c/Users/me/Documents/Clone Hero/Songs" --out ../data
  python process_batch.py --source /mnt/d/songs --out ../data --limit 3
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Fix print encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from chart_parser import parse_song, parse_ini
from tokenizer import tokenize_events, save_tokens
from sng_parser import extract_sng

MANIFEST_COLUMNS = [
    'song_id', 'title', 'artist', 'source_folder', 'duration_s', 'has_medium',
    'n_notes_medium', 'min_ioi_ms', 'collision_100ms', 'status', 'error_msg',
    'alignment_suspect', 'outlier_length',
]


def load_manifest(manifest_path):
    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)
        for col in MANIFEST_COLUMNS:
            if col not in df.columns:
                df[col] = ''
        return df
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def get_song_id(folder_name):
    return "".join(x for x in folder_name if x.isalnum())


def collect_song_folders(source_dir, out_dir):
    """Returns song folders from source_dir, extracting any .sng files first."""
    source = Path(source_dir)
    extract_root = Path(out_dir) / 'extracted'
    folders = []
    for entry in sorted(source.iterdir()):
        if entry.is_dir():
            folders.append(entry)
        elif entry.suffix.lower() == '.sng':
            dest = extract_root / entry.stem
            if not (dest / 'song.ini').exists() and not any(dest.glob('notes.*')):
                try:
                    meta = extract_sng(str(entry), str(dest))
                except Exception as e:
                    print(f"[!] Failed to extract {entry.name}: {e}")
                    continue
                # .sng keeps song.ini fields in its metadata block; write them
                # out so parse_ini / parse_song can read delay, name, artist.
                if meta and not (dest / 'song.ini').exists():
                    with open(dest / 'song.ini', 'w', encoding='utf-8') as f:
                        f.write('[song]\n')
                        for k, v in meta.items():
                            f.write(f'{k} = {v}\n')
            folders.append(dest)
    return folders


def process_folder(folder, out_dir, encoder):
    """Processes one song folder. Returns a manifest row dict."""
    folder_name = folder.name
    song_id = get_song_id(folder_name)
    ini_data = parse_ini(str(folder / 'song.ini'))

    row = {
        'song_id': song_id,
        'title': ini_data.get('name', 'Unknown'),
        'artist': ini_data.get('artist', 'Unknown'),
        'source_folder': folder_name,
        'duration_s': 0.0,
        'has_medium': False,
        'n_notes_medium': 0,
        'min_ioi_ms': 0.0,
        'collision_100ms': False,
        'status': 'done',
        'error_msg': '',
        'alignment_suspect': False,
        'outlier_length': False,
    }

    try:
        events = parse_song(str(folder))
        if events is None:
            row.update(status='skipped', error_msg='No Medium guitar track found')
            return row

        row['has_medium'] = True
        row['n_notes_medium'] = len(events)
        if len(events) > 1:
            iois = [events[i + 1][0] - events[i][0] for i in range(len(events) - 1)]
            row['min_ioi_ms'] = min(iois) * 1000.0

        tokens, sustains, raw_events, collision = tokenize_events(events)
        row['collision_100ms'] = collision

        if encoder is not None:
            codes, duration_s = encoder.process_song(str(folder), out_dir, song_id)
            if codes is None:
                row.update(status='error', error_msg='Audio processing failed')
                return row
            row['duration_s'] = duration_s
            row['outlier_length'] = duration_s > 600  # 10 minutes
            last_note_time = events[-1][0]
            row['alignment_suspect'] = (
                last_note_time > duration_s or (duration_s - last_note_time) > 60
            )

        else:
            # Not 'done': a later run with audio enabled will pick it up again.
            row['status'] = 'tokens_only'

        save_tokens(out_dir, song_id, tokens, sustains, raw_events)
    except Exception as e:
        row.update(status='error', error_msg=str(e))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='Folder of song folders and/or .sng files')
    parser.add_argument('--out', default='data', help='Output folder (default: ./data)')
    parser.add_argument('--limit', type=int, default=None, help='Process at most N new songs')
    parser.add_argument('--no-audio', action='store_true',
                        help='Skip EnCodec audio encoding (charts/tokens only, no torch needed)')
    parser.add_argument('--reprocess', action='store_true',
                        help='Re-run songs already marked done in the manifest')
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, 'manifest.csv')
    manifest = load_manifest(manifest_path)
    done_ids = set() if args.reprocess else set(manifest[manifest['status'] == 'done']['song_id'])

    encoder = None
    if not args.no_audio:
        from audio_processor import AudioEncoder  # heavy: torch + encodec
        encoder = AudioEncoder()

    folders = collect_song_folders(args.source, out_dir)
    print(f"Found {len(folders)} songs in {args.source} ({len(done_ids)} already done).")

    processed = 0
    for folder in folders:
        if args.limit is not None and processed >= args.limit:
            break
        song_id = get_song_id(folder.name)
        if song_id in done_ids:
            continue

        print(f"Processing {folder.name}...")
        row = process_folder(folder, out_dir, encoder)
        print(f"  -> {row['status']}: {row['n_notes_medium']} notes"
              + (f" ({row['error_msg']})" if row['error_msg'] else ''))

        # Replace any previous row for this song, then save after every song
        manifest = manifest[manifest['song_id'] != song_id]
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
        manifest.to_csv(manifest_path, index=False)
        processed += 1

    print(f"Done. Processed {processed} songs. Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
