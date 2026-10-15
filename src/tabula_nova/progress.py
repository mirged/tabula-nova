"""Terminal progress without changing machine-readable stdout or GPU execution."""

import os
import sys

from tqdm import tqdm


def progress_enabled():
    setting = os.environ.get('TABULA_PROGRESS', 'auto').strip().lower()
    if setting in ('1', 'true', 'yes', 'on'):
        return True
    if setting in ('0', 'false', 'no', 'off'):
        return False
    return bool(getattr(sys.stderr, 'isatty', lambda: False)())


def progress(iterable=None, **options):
    options.setdefault('disable', not progress_enabled())
    options.setdefault('file', sys.stderr)
    options.setdefault('dynamic_ncols', True)
    options.setdefault('mininterval', 0.5)
    options.setdefault('delay', 0.5)
    return tqdm(iterable, **options)


def progress_write(message):
    # Clear/repaint any active bars around a normal stdout status message.
    tqdm.write(message, file=sys.stdout)
    sys.stdout.flush()
