"""Train/val/test split over the processed manifest, grouped by *song*.

Several charts of the same song (e.g. "Misery Business (Harmonix)" and
"Misery Business (Neversoft)") share the same audio, so they must land in the
same split or the model is tested on music it trained on. Charts are grouped by
normalised artist + title (charter tags in (...) / [...] stripped, "ft." guests
dropped); mashups like "A vs. B" are joined to the groups of both A and B.

Assignments are sticky: groups already in the split file keep their split, and
new charts of an existing song join that song's split.

Usage:
  python split.py --data ../data            # reads/writes ../data/dataset_split.json
  python split.py --data ../data --include-tokens-only
"""
import argparse
import json
import os
import random
import re

import pandas as pd

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def _norm(text):
    text = str(text).lower()
    text = re.sub(r'[\(\[].*?[\)\]]', ' ', text)        # (Harmonix), [md5], (OSU! edit)
    return re.sub(r'[^a-z0-9]+', '', text)


def _norm_artist(artist):
    artist = re.split(r'\s+(?:ft\.?|feat\.?|featuring)\s+', str(artist).lower())[0]
    return _norm(artist)


def song_keys(artist, title):
    """All group keys this chart belongs to (more than one for mashups)."""
    a = _norm_artist(artist)
    parts = re.split(r'\s+vs\.?\s+|\s*[/／]\s*', str(title).lower())
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
    """rows: list of (song_id, artist, title). Returns list of song_id groups."""
    uf = _UnionFind()
    for song_id, artist, title in rows:
        node = f'id:{song_id}'
        for key in song_keys(artist, title):
            uf.union(node, f'key:{key}')
    groups = {}
    for song_id, _, _ in rows:
        groups.setdefault(uf.find(f'id:{song_id}'), []).append(song_id)
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='data', help='Folder with manifest.csv (output of process_batch.py)')
    parser.add_argument('--include-tokens-only', action='store_true',
                        help='Also split songs processed with --no-audio')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    manifest = pd.read_csv(os.path.join(args.data, 'manifest.csv'))
    ok = {'done', 'tokens_only'} if args.include_tokens_only else {'done'}
    manifest = manifest[manifest['status'].isin(ok) & (manifest['n_notes_medium'] > 0)]
    rows = list(zip(manifest['song_id'], manifest['artist'], manifest['title']))

    split_path = os.path.join(args.data, 'dataset_split.json')
    split = {'train': [], 'val': [], 'test': []}
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
    assigned = {sid: name for name, ids in split.items() for sid in ids}

    rng = random.Random(args.seed)
    new_groups = []
    for group in group_songs(rows):
        existing = {assigned[s] for s in group if s in assigned}
        if len(existing) > 1:
            print(f"[!] Leak in existing split, same song in {sorted(existing)}: {group}")
        if existing:
            target = sorted(existing)[0]
            split[target].extend(s for s in group if s not in assigned)
        else:
            new_groups.append(group)

    rng.shuffle(new_groups)
    n = len(new_groups)
    train_end = int(round(n * TRAIN_RATIO))
    val_end = train_end + int(round(n * VAL_RATIO))
    for i, group in enumerate(new_groups):
        name = 'train' if i < train_end else 'val' if i < val_end else 'test'
        split[name].extend(group)

    with open(split_path, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"{len(rows)} charts in {len(group_songs(rows))} song groups ({n} new groups).")
    for name, ids in split.items():
        print(f"  {name}: {len(ids)} charts")
    print(f"Saved {split_path}")


if __name__ == '__main__':
    main()
