import re
from pathlib import Path

def glob_re(path, regex="", glob_mask="*", inverse=False):
    p = Path(path)
    if inverse:
        res = [str(f) for f in p.glob(glob_mask) if not re.search(regex, str(f))]
    else:
        res = [str(f) for f in p.glob(glob_mask) if re.search(regex, str(f))]
    return res