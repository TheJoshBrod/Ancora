DIFFICULTIES = ('easy', 'medium', 'hard', 'expert')
DIFF_ABBR = {'easy': 'E', 'medium': 'M', 'hard': 'H', 'expert': 'X'}

# fret bitmask: bits 0-4 = G R Y B O, bit 5 = open
OPEN_BIT = 1 << 5

FLAG_FORCED = 1  # .chart N 5: inverts the natural HOPO/strum state
FLAG_FORCE_HOPO = 2
FLAG_FORCE_STRUM = 4
FLAG_TAP = 8
FLAG_STAR_POWER = 16

CHART_SECTIONS = {
    'EasySingle': 'easy',
    'MediumSingle': 'medium',
    'HardSingle': 'hard',
    'ExpertSingle': 'expert',
}

MIDI_BASE = {'easy': 60, 'medium': 72, 'hard': 84, 'expert': 96}
MIDI_TAP_NOTE = 104
MIDI_STAR_POWER_NOTE = 116
PS_SYSEX_DIFF = {0: 'easy', 1: 'medium', 2: 'hard', 3: 'expert'}
PS_SYSEX_OPEN, PS_SYSEX_TAP = 1, 4
