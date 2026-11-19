# sequence: BOS BEAT [POS_k FRETS_m]* BEAT ... EOS, k on a 1/SUBDIV-beat grid;
# sustain classes and flags are side arrays aligned with the tokens

import numpy as np

SUBDIV = 48

PAD, BOS, EOS, BEAT = 0, 1, 2, 3
POS_OFFSET = 4
FRETS_OFFSET = POS_OFFSET + SUBDIV
VOCAB_SIZE = FRETS_OFFSET + 63

SUSTAIN_CLASSES = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8], dtype=np.float64)
# Guess at Clone Hero's cutoff; song.ini sustain_cutoff_threshold overrides it
DEFAULT_SUSTAIN_CUTOFF_BEATS = 0.25
SNAP_TOLERANCE = 0.1  # in grid steps
