import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ancora.data.audio import AudioEncoder
from ancora.data.chart import parse_ini, parse_song
from ancora.data.sng import extract_sng
from ancora.data.tokenizer import save_tokens, tokenize_events

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    'song_id',
    'title',
    'artist',
    'source_folder',
    'duration_s',
    'has_medium',
    'n_notes_medium',
    'min_ioi_ms',
    'collision_100ms',
    'status',
    'error_msg',
    'alignment_suspect',
    'outlier_length',
]


def load_manifest(manifest_path):
    if Path(manifest_path).exists():
        df = pd.read_csv(manifest_path)
        for col in MANIFEST_COLUMNS:
            if col not in df.columns:
                df[col] = ''
        return df
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def get_song_id(folder_name):
    return "".join(x for x in folder_name if x.isalnum())


def collect_song_folders(source_dir, out_dir):
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
                    logger.warning("Failed to extract %s: %s", entry.name, e)
                    continue
                # .sng keeps song.ini fields in its metadata block; write them
                # out so parse_ini / parse_song can read delay, name, artist.
                if meta and not (dest / 'song.ini').exists():
                    with (dest / 'song.ini').open('w', encoding='utf-8') as f:
                        f.write('[song]\n')
                        for k, v in meta.items():
                            f.write(f'{k} = {v}\n')
            folders.append(dest)
    return folders


def process_folder(folder, out_dir, encoder):
    folder_name = folder.name
    song_id = get_song_id(folder_name)
    ini_data = parse_ini(str(folder / 'song.ini'))

    row: dict[str, Any] = {
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


def run(source, out_dir, limit=None, no_audio=False, reprocess=False):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / 'manifest.csv'
    manifest = load_manifest(manifest_path)
    done_ids = set() if reprocess else set(manifest[manifest['status'] == 'done']['song_id'])

    encoder = None if no_audio else AudioEncoder()

    folders = collect_song_folders(source, out_dir)
    logger.info("Found %d songs in %s (%d already done).", len(folders), source, len(done_ids))

    processed = 0
    for folder in folders:
        if limit is not None and processed >= limit:
            break
        song_id = get_song_id(folder.name)
        if song_id in done_ids:
            continue

        logger.info("Processing %s...", folder.name)
        row = process_folder(folder, out_dir, encoder)
        logger.info(
            "  -> %s: %s notes%s",
            row['status'],
            row['n_notes_medium'],
            f" ({row['error_msg']})" if row['error_msg'] else '',
        )

        # Replace any previous row for this song, then save after every song
        manifest = manifest.loc[manifest['song_id'] != song_id]
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
        manifest.to_csv(manifest_path, index=False)
        processed += 1

    logger.info("Done. Processed %d songs. Manifest: %s", processed, manifest_path)
