# Charts of the same song share audio, so they must land in the same split or the
# model is tested on music it trained on. Assignments are sticky: groups already in
# the split file keep their split, and new charts of a known song join it.

import json
import logging
import random
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def _norm(text):
    text = str(text).lower()
    text = re.sub(r'[\(\[].*?[\)\]]', ' ', text)  # (Harmonix), [md5], (OSU! edit)
    return re.sub(r'[^a-z0-9]+', '', text)


def _norm_artist(artist):
    artist = re.split(r'\s+(?:ft\.?|feat\.?|featuring)\s+', str(artist).lower())[0]
    return _norm(artist)


def song_keys(artist, title):
    # mashups ("A vs B", "A / B") belong to more than one group
    a = _norm_artist(artist)
    parts = re.split(r'\s+vs\.?\s+|\s*[/／]\s*', str(title).lower())  # noqa: RUF001 (fullwidth slash in titles)
    keys = {f'{a}|{_norm(title)}'}
    if len(parts) > 1:
        keys |= {f'{a}|{_norm(p)}' for p in parts if _norm(p)}
    return keys


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def group_songs(rows):
    uf = _UnionFind()
    for song_id, artist, title in rows:
        node = f'id:{song_id}'
        for key in song_keys(artist, title):
            uf.union(node, f'key:{key}')
    groups = {}
    for song_id, _, _ in rows:
        groups.setdefault(uf.find(f'id:{song_id}'), []).append(song_id)
    return list(groups.values())


def find_leaks(split, rows):
    assigned = {sid: name for name, ids in split.items() for sid in ids}
    leaks = []
    for group in group_songs(rows):
        names = sorted({assigned[s] for s in group if s in assigned})
        if len(names) > 1:
            leaks.append((group, names))
    dupes = [sid for sid in assigned if sum(sid in ids for ids in split.values()) > 1]
    if dupes:
        leaks.append((dupes, ['listed in several splits']))
    return leaks


def run(data_dir, check=False, seed=0):
    data_dir = Path(data_dir)
    manifest = pd.read_csv(data_dir / 'manifest.csv')
    manifest = manifest[(manifest['status'] == 'done') & (manifest['n_notes_total'] > 0)]
    rows = list(zip(manifest['song_id'], manifest['artist'], manifest['title'], strict=True))

    split_path = data_dir / 'dataset_split.json'
    split = {'train': [], 'val': [], 'test': []}
    if split_path.exists():
        with split_path.open() as f:
            split = json.load(f)
    if check:
        leaks = find_leaks(split, rows)
        for group, names in leaks:
            logger.warning("Leak: %s in %s", group, names)
        logger.info('No leaks.' if not leaks else f'{len(leaks)} leak(s).')
        return 1 if leaks else 0
    assigned = {sid: name for name, ids in split.items() for sid in ids}

    rng = random.Random(seed)
    new_groups = []
    for group in group_songs(rows):
        existing = {assigned[s] for s in group if s in assigned}
        if len(existing) > 1:
            logger.warning("Leak in existing split, same song in %s: %s", sorted(existing), group)
        if existing:
            target = sorted(existing)[0]
            split[target].extend(s for s in group if s not in assigned)
        else:
            new_groups.append(group)

    rng.shuffle(new_groups)
    n = len(new_groups)
    train_end = round(n * TRAIN_RATIO)
    val_end = train_end + round(n * VAL_RATIO)
    for i, group in enumerate(new_groups):
        name = 'train' if i < train_end else 'val' if i < val_end else 'test'
        split[name].extend(group)

    with split_path.open('w') as f:
        json.dump(split, f, indent=2)
    for group, names in find_leaks(split, rows):
        logger.warning("Leak after splitting: %s in %s", group, names)

    logger.info(
        "%d charts in %d song groups (%d new groups).", len(rows), len(group_songs(rows)), n
    )
    for name, ids in split.items():
        logger.info("  %s: %d charts", name, len(ids))
    logger.info("Saved %s", split_path)
    return 0
