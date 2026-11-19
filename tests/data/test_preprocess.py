import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from test_factories import write_audio, write_chart

from ancora.data import preprocess as pp
from ancora.data.audio.encoders.mel import BeatSyncMel
from ancora.data.tokenizer.services import subdivision_times


def test_process_folder():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / 'src'
        for name in ('Tester - Test Song (Me)', 'Tester - Test Song (Someone Else)'):
            write_chart(src / name)
            write_audio(src / name)
        out = Path(tmp) / 'out'
        catalog, locations = pp.build_catalog([str(src)], str(out))
        assert len(catalog) == 2 and (catalog['duplicate_of'] != '').sum() == 1
        sid = catalog.loc[catalog['duplicate_of'] == '', 'song_id'].iloc[0]
        folder, source_dir = locations[sid]
        mel_encoder = BeatSyncMel('cpu')
        row = pp.process_folder(folder, source_dir, sid, str(out), [mel_encoder], False)
        assert row['status'] == 'done', row
        assert (
            row['difficulties'] == 'MX'
            and row['n_notes_medium'] == 5
            and row['n_notes_expert'] == 3
        )
        assert np.isclose(row['duration_s'], 4.0)
        song_dir = out / 'tokens' / sid
        tok = np.load(song_dir / 'tokens.npz')
        assert {'medium_chart_tokens', 'expert_chart_sustain'} <= set(tok.files)
        raw = np.load(song_dir / 'chart.npz')
        assert list(raw['section_names']) == ['Intro', 'Chorus 1']
        mel = np.load(song_dir / 'mel_beat48.npy')
        assert mel.shape == (len(subdivision_times(raw['beats_chart'])), mel_encoder.n_mels)


def test_catalog_primary_follows_source_order(tmp_path):
    official, chorus = tmp_path / 'official', tmp_path / 'chorus'
    broken = official / 'Tester - Test Song (Broken)'
    write_chart(broken)
    (broken / 'notes.chart').unlink()
    (broken / 'notes.mid').write_bytes(b'MThd\x00\x00\x00\x06\x00\x01')
    write_chart(official / 'Tester - Test Song (Official)')
    write_chart(chorus / 'Tester - Test Song (A Fan)')
    catalog, _ = pp.build_catalog([str(chorus), str(official)], str(tmp_path / 'out'))
    assert catalog.loc[catalog['duplicate_of'] == '', 'folder'].tolist() == [
        'Tester - Test Song (A Fan)'
    ]
    catalog, _ = pp.build_catalog([str(official), str(chorus)], str(tmp_path / 'out'))
    assert catalog.loc[catalog['duplicate_of'] == '', 'folder'].tolist() == [
        'Tester - Test Song (Official)'
    ]


def test_run_processes_every_copy(tmp_path):
    for name in ('Tester - Test Song (Me)', 'Tester - Test Song (Someone Else)'):
        write_chart(tmp_path / 'src' / name)
        write_audio(tmp_path / 'src' / name)
    pp.run([str(tmp_path / 'src')], tmp_path / 'out', features=['mel'])
    manifest = pd.read_csv(tmp_path / 'out' / 'manifest.csv', keep_default_na=False)
    assert list(manifest['status']) == ['done', 'done']
    assert (manifest['duplicate_of'] != '').sum() == 1
    pp.run([str(tmp_path / 'src')], tmp_path / 'out2', features=['mel'], primary_only=True)
    assert len(pd.read_csv(tmp_path / 'out2' / 'manifest.csv')) == 1


def test_run_writes_vocab_config(tmp_path):
    write_chart(tmp_path / 'src' / 'song')
    pp.run([str(tmp_path / 'src')], tmp_path / 'out', features=['mel'], limit=0)
    assert (tmp_path / 'out' / 'tokenizer_config.json').exists()
