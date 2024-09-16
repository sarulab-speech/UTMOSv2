import os
from pathlib import Path

_UTMOSV2_CHACHE = Path(os.getenv("UTMOSV2_CHACHE", "~/.cache/utmosv2")).expanduser()
