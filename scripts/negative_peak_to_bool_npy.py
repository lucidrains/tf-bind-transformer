#/usr/bin/python

import polars as pl
import numpy as np
from pathlib import Path
import sys

NEGATIVE_PEAK_PATH = sys.argv[1]
NUMROWS = int(sys.argv[2])
ID_COLUMN = 'column_6'

df = pl.read_csv(NEGATIVE_PEAK_PATH, sep = '\t', has_headers = False)
np_array = df.get_column(ID_COLUMN).to_numpy()

to_save = np.full((NUMROWS,), False)
to_save[np_array - 1] = True

p = Path(NEGATIVE_PEAK_PATH)
filename = f'{p.stem}.bool'
np.save(filename, to_save)

print(f'{filename} saved')
