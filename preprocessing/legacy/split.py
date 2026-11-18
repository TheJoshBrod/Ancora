import os
import json
import random

SPLIT_FILE = "songs/dataset_split.json"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def load_existing_split():
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, "r") as f:
            return json.load(f)
    return {"train": [], "val": [], "test": []}

def save_split(split):
    os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump(split, f, indent=4)

def build_sample(base_name):
    """Builds the sample dict based on the common file prefix."""
    return {
        "song_id": base_name,
        "audio": f"{base_name}.wav",
        "notes": f"{base_name}.json"
    }

def main():
    dataset_dir = "songs/"

    # 1. Find all unique songs by looking at the .json files
    base_names = []
    for f in os.listdir(dataset_dir):
        if f.endswith('.json') and f != "dataset_split.json":
            # Strip the '.json' extension to get the base name
            base_names.append(f[:-5])

    print(f"Found {len(base_names)} songs in directory.")

    split = load_existing_split()

    # 2. Track what has already been processed using the new 'song_id' key
    existing_songs = set()
    for category in ["train", "val", "test"]:
        for sample in split[category]:
            existing_songs.add(sample.get("song_id"))

    # 3. Filter out songs already in the manifest
    new_songs = [b for b in base_names if b not in existing_songs]

    if not new_songs:
        print("No new songs to assign.")
        return

    random.shuffle(new_songs)

    n = len(new_songs)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    # 4. Distribute the flat files into the manifest
    for b in new_songs[:train_end]:
        split["train"].append(build_sample(b))

    for b in new_songs[train_end:val_end]:
        split["val"].append(build_sample(b))

    for b in new_songs[val_end:]:
        split["test"].append(build_sample(b))

    save_split(split)

    print(f"\nAdded {n} new songs to the manifest.")
    print(f"Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}")
    print(f"Manifest saved to {SPLIT_FILE}")

if __name__ == "__main__":
    main()