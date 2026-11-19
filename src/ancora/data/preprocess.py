import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ancora.data.audio.constants import AUDIO_EXTS, SR
from ancora.data.audio.services import build_encoders, load_song_audio, save_features
from ancora.data.chart.constants import DIFF_ABBR, DIFFICULTIES
from ancora.data.chart.services import load_chart, parse_ini
from ancora.data.sng import extract_sng
from ancora.data.split import song_keys
from ancora.data.tokenizer.constants import DEFAULT_SUSTAIN_CUTOFF_BEATS
from ancora.data.tokenizer.services import (
    save_tokens,
    subdivision_times,
    tokenize,
    write_vocab_config,
)

logger = logging.getLogger(__name__)

OFFICIAL_CHARTERS = frozenset(
    {'harmonix', 'neversoft', 'vicarious visions', 'freestylegames', 'budcat', 'beenox'}
)

MANIFEST_COLUMNS = (
    [
        'song_id',
        'title',
        'artist',
        'charter',
        'source_dir',
        'source_folder',
        'chart_format',
        'resolution',
        'difficulties',
        'n_notes_total',
    ]
    + [f'n_notes_{d}' for d in DIFFICULTIES]
    + [f'offgrid_{d}' for d in DIFFICULTIES]
    + [
        'collisions',
        'n_beats',
        'min_ioi_ms',
        'duration_s',
        'has_guitar_stem',
        'n_sections',
        'status',
        'error_msg',
        'alignment_suspect',
        'outlier_length',
    ]
)


def load_manifest(manifest_path):
    if Path(manifest_path).exists():
        df = pd.read_csv(manifest_path)
        for col in MANIFEST_COLUMNS:
            if col not in df.columns:
                df[col] = ''
        return df
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def collect_song_folders(source_dir, out_dir):
    source = Path(source_dir)
    extract_root = Path(out_dir) / 'extracted'
    found = []
    for entry in sorted(source.iterdir()):
        if entry.is_dir():
            found.append((entry, 'folder'))
        elif entry.suffix.lower() == '.sng':
            dest = extract_root / entry.stem
            if not (dest / 'song.ini').exists() and not any(dest.glob('notes.*')):
                try:
                    meta = extract_sng(str(entry), str(dest))
                except Exception as e:
                    logger.warning("Failed to extract %s: %s", entry.name, e)
                    continue
                # .sng stores song.ini fields in its metadata block
                if meta and not (dest / 'song.ini').exists():
                    with (dest / 'song.ini').open('w', encoding='utf-8') as f:
                        f.write('[song]\n')
                        for k, v in meta.items():
                            f.write(f'{k} = {v}\n')
            found.append((dest, 'sng'))
    return found


def make_song_ids(folders):
    base = ["".join(c for c in f.name if c.isalnum()) for f in folders]
    counts = Counter(base)
    ids = []
    for f, b in zip(folders, base, strict=True):
        if counts[b] > 1:
            b = f"{b}_{hashlib.md5(str(f).encode()).hexdigest()[:6]}"
        ids.append(b)
    return ids


def catalog_row(folder, kind, source_dir, song_id):
    ini = parse_ini(str(folder / 'song.ini'))
    audio = [
        p
        for p in folder.iterdir()
        if p.suffix.lower() in AUDIO_EXTS and not p.name.lower().startswith(('preview', 'crowd'))
    ]
    fmt = (
        'chart'
        if (folder / 'notes.chart').exists()
        else 'mid'
        if (folder / 'notes.mid').exists()
        else ''
    )
    diffs, error = '', ''
    try:
        chart = load_chart(str(folder))
        if chart is not None:
            diffs = ''.join(DIFF_ABBR[d] for d in chart.difficulties_present())
    except Exception as e:
        error = str(e)
    artist, title = ini.get('artist', 'Unknown'), ini.get('name', 'Unknown')
    return {
        'song_id': song_id,
        'source_dir': str(source_dir),
        'folder': folder.name,
        'kind': kind,
        'artist': artist,
        'title': title,
        'charter': ini.get('charter', ini.get('frets', '')),
        'chart_format': fmt,
        'difficulties': diffs,
        'n_audio_files': len(audio),
        'has_guitar_stem': len(audio) > 1 and any(p.stem.lower() == 'guitar' for p in audio),
        'group_key': sorted(song_keys(artist, title))[0],
        'duplicate_of': '',
        'error_msg': error,
    }


def build_catalog(sources, out_dir):
    entries = []
    for src in sources:
        for folder, kind in collect_song_folders(src, out_dir):
            entries.append((folder, kind, src))
    ids = make_song_ids([f for f, _, _ in entries])
    rows = [catalog_row(f, k, s, sid) for (f, k, s), sid in zip(entries, ids, strict=True)]
    catalog = pd.DataFrame(rows)
    kept = {}
    for i in sorted(range(len(rows)), key=lambda i: _keep_rank(rows[i])):
        key = rows[i]['group_key']
        if key in kept:
            catalog.at[i, 'duplicate_of'] = kept[key]
        else:
            kept[key] = rows[i]['song_id']
    return catalog, {sid: (f, s) for (f, _, s), sid in zip(entries, ids, strict=True)}


def _keep_rank(row):
    readable = row['error_msg'] == '' and row['difficulties'] != ''
    charter = re.sub(r'<[^>]*>', '', row['charter']).strip().lower()
    return (not readable, charter not in OFFICIAL_CHARTERS)


def _sustain_cutoff_beats(chart):
    raw = chart.ini.get('sustain_cutoff_threshold', '')
    try:
        ticks = float(raw)
        if ticks > 0:
            return ticks / chart.resolution
    except ValueError:
        pass
    return DEFAULT_SUSTAIN_CUTOFF_BEATS


def process_folder(folder, source_dir, song_id, out_dir, encoders, guitar_stem):
    ini = parse_ini(str(folder / 'song.ini'))
    row: dict[str, Any] = {c: '' for c in MANIFEST_COLUMNS}
    row.update(
        song_id=song_id,
        title=ini.get('name', 'Unknown'),
        artist=ini.get('artist', 'Unknown'),
        charter=ini.get('charter', ini.get('frets', '')),
        source_dir=str(source_dir),
        source_folder=folder.name,
        status='done',
        n_notes_total=0,
        collisions=0,
        duration_s=0.0,
        min_ioi_ms=0.0,
        has_guitar_stem=False,
        alignment_suspect=False,
        outlier_length=False,
    )
    for d in DIFFICULTIES:
        row[f'n_notes_{d}'] = 0
        row[f'offgrid_{d}'] = 0
    try:
        chart = load_chart(str(folder))
        if chart is None:
            row.update(status='skipped', error_msg='No notes.chart or notes.mid')
            return row
        present = chart.difficulties_present()
        row.update(
            chart_format=chart.fmt,
            resolution=chart.resolution,
            difficulties=''.join(DIFF_ABBR[d] for d in present),
            n_sections=len(chart.sections),
        )
        if not present:
            row.update(status='skipped', error_msg='No lead guitar notes')
            return row

        song_dir = Path(out_dir) / 'tokens' / song_id
        beats_chart = chart.beat_times()
        cutoff = _sustain_cutoff_beats(chart)
        raw, tok = {}, {}
        raw['beats_chart'] = beats_chart
        raw['downbeats_chart_idx'] = chart.downbeat_indices(len(beats_chart))
        raw['resolution'] = np.array(chart.resolution)
        raw['section_ticks'] = np.array([t for t, _ in chart.sections], dtype=np.int64)
        raw['section_names'] = np.array([n for _, n in chart.sections], dtype=str)

        last_note_s, top_diff_times = 0.0, None
        for d in present:
            notes = chart.notes[d]
            ticks = np.array([n.tick for n in notes], dtype=np.int64)
            lengths = np.array([n.length for n in notes], dtype=np.int64)
            frets = np.array([n.frets for n in notes], dtype=np.int64)
            flags = np.array([n.flags for n in notes], dtype=np.int64)
            times = chart.ticks_to_seconds(ticks)
            raw.update(
                {
                    f'{d}_tick': ticks,
                    f'{d}_length': lengths,
                    f'{d}_frets': frets.astype(np.uint8),
                    f'{d}_flags': flags.astype(np.uint8),
                    f'{d}_time_s': times,
                    f'{d}_end_s': chart.ticks_to_seconds(ticks + lengths),
                    f'{d}_star_power': np.array(
                        chart.star_power.get(d, []), dtype=np.int64
                    ).reshape(-1, 2),
                }
            )
            res = tokenize(
                ticks / chart.resolution,
                (ticks + lengths) / chart.resolution,
                frets,
                flags,
                n_beats=len(beats_chart),
                sustain_cutoff=cutoff,
            )
            tok[f'{d}_chart_tokens'] = res.tokens
            tok[f'{d}_chart_sustain'] = res.sustain
            tok[f'{d}_chart_flags'] = res.flags
            row[f'n_notes_{d}'] = res.stats.n_notes
            row[f'offgrid_{d}'] = res.stats.n_offgrid
            row['collisions'] += res.stats.n_collisions
            row['n_notes_total'] += res.stats.n_notes
            last_note_s = max(last_note_s, float(times.max()))
            top_diff_times = times
        if top_diff_times is not None and len(top_diff_times) > 1:
            row['min_ioi_ms'] = (
                float(np.diff(np.unique(top_diff_times)).min() * 1000.0)
                if len(np.unique(top_diff_times)) > 1
                else 0.0
            )
        row['n_beats'] = len(beats_chart)

        mix, guitar = load_song_audio(str(folder), guitar_stem=guitar_stem)
        if mix is None:
            row.update(status='error', error_msg='Audio loading failed')
            return row
        row['duration_s'] = mix.shape[1] / SR
        row['outlier_length'] = row['duration_s'] > 600
        row['alignment_suspect'] = (
            last_note_s > row['duration_s'] or (row['duration_s'] - last_note_s) > 60
        )

        sub_times = subdivision_times(beats_chart)
        save_features(song_dir, encoders, mix, sub_times)
        if guitar is not None:
            save_features(song_dir, encoders, guitar, sub_times, suffix='_guitar')
            row['has_guitar_stem'] = True

        song_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(song_dir / 'chart.npz', **raw)
        save_tokens(out_dir, song_id, tok)
    except Exception as e:
        row.update(status='error', error_msg=f'{type(e).__name__}: {e}')
    return row


def run(
    sources,
    out_dir,
    catalog_only=False,
    limit=None,
    features=('encodec',),
    guitar_stem=False,
    include_duplicates=False,
    reprocess=False,
):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog, locations = build_catalog(sources, out_dir)
    catalog.to_csv(out_dir / 'catalog.csv', index=False)
    n_dup = int((catalog['duplicate_of'] != '').to_numpy().sum()) if len(catalog) else 0
    logger.info(
        "Catalog: %d charts from %d source(s), %d duplicates -> %s",
        len(catalog),
        len(sources),
        n_dup,
        out_dir / 'catalog.csv',
    )
    if catalog_only:
        return

    write_vocab_config(out_dir)
    manifest_path = out_dir / 'manifest.csv'
    manifest = load_manifest(manifest_path)
    done_ids = set() if reprocess else set(manifest[manifest['status'] == 'done']['song_id'])
    encoders = build_encoders(features)

    todo = catalog if include_duplicates else catalog[catalog['duplicate_of'] == '']
    processed = 0
    for song_id in todo['song_id']:
        if limit is not None and processed >= limit:
            break
        if song_id in done_ids:
            continue
        folder, source_dir = locations[song_id]
        logger.info("Processing %s...", folder.name)
        row = process_folder(folder, source_dir, song_id, out_dir, encoders, guitar_stem)
        logger.info(
            "  -> %s: %s notes [%s]%s",
            row['status'],
            row['n_notes_total'],
            row['difficulties'],
            f" ({row['error_msg']})" if row['error_msg'] else '',
        )
        manifest = manifest.loc[manifest['song_id'] != song_id]
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
        manifest.to_csv(manifest_path, index=False)
        processed += 1

    logger.info("Done. Processed %d songs. Manifest: %s", processed, manifest_path)
