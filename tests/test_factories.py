import wave

import mido
import numpy as np

from ancora.data.audio.constants import SR

CHART = """[Song]
{
  Name = "Test Song"
  Resolution = 192
  Offset = 0
}
[SyncTrack]
{
  0 = TS 4
  0 = B 120000
  768 = B 60000
}
[Events]
{
  0 = E "section Intro"
  768 = E "section Chorus 1"
}
[MediumSingle]
{
  0 = N 0 0
  192 = N 1 0
  192 = N 2 96
  384 = N 7 0
  576 = N 3 384
  576 = N 5 0
  768 = N 4 0
  768 = N 6 0
  768 = S 2 192
}
[ExpertSingle]
{
  0 = N 0 0
  64 = N 1 0
  128 = N 2 0
}
"""


def write_chart(folder, ini_delay=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'notes.chart').write_text(CHART)
    ini = '[song]\nname = Test Song\nartist = Tester\ncharter = Me\n'
    if ini_delay is not None:
        ini += f'delay = {ini_delay}\n'
    (folder / 'song.ini').write_text(ini)


def write_audio(folder, seconds=4.0):
    t = np.arange(int(SR * seconds)) / SR
    pcm = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype('<i2')
    with wave.open(str(folder / 'song.wav'), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def write_mid(path):
    mid = mido.MidiFile(ticks_per_beat=480)
    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage('track_name', name='tempo', time=0))
    tempo.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    tempo.append(mido.MetaMessage('time_signature', numerator=3, denominator=4, time=0))
    mid.tracks.append(tempo)

    events = mido.MidiTrack()
    events.append(mido.MetaMessage('track_name', name='EVENTS', time=0))
    events.append(mido.MetaMessage('text', text='[section verse_1]', time=0))
    events.append(mido.MetaMessage('text', text='[prc_chorus_1]', time=960))
    mid.tracks.append(events)

    g = mido.MidiTrack()
    g.append(mido.MetaMessage('track_name', name='PART GUITAR', time=0))
    g.append(mido.MetaMessage('text', text='[ENHANCED_OPENS]', time=0))
    ev = [
        (0, 'on', 72),
        (0, 'on', 73),
        (0, 'on', 77),
        (120, 'off', 72),
        (120, 'off', 73),
        (120, 'off', 77),
        (480, 'on', 71),
        (600, 'off', 71),
        (960, 'on', 104),
        (960, 'on', 116),
        (960, 'on', 74),
        (1440, 'off', 74),
        (1440, 'off', 104),
        (1440, 'off', 116),
        (0, 'on', 96),
        (240, 'off', 96),
    ]
    ev.sort(key=lambda e: (e[0], e[1] != 'off'))
    last = 0
    for tick, kind, note in ev:
        g.append(
            mido.Message(
                'note_on', note=note, velocity=100 if kind == 'on' else 0, time=tick - last
            )
        )
        last = tick
    mid.tracks.append(g)
    mid.save(path)
