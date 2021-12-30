import os
from shutil import rmtree
import torch
import hashlib
from functools import wraps
from pathlib import Path

# constants

CACHE_PATH = Path(os.getenv('CACHE_PATH', os.path.expanduser('~/.cache.tf.bind.transformer')))
CACHE_PATH.mkdir(exist_ok = True, parents = True)

CLEAR_CACHE = os.getenv('CLEAR_CACHE', None) is not None
VERBOSE = os.getenv('VERBOSE', None) is not None

# helper functions

def log(s):
    if not VERBOSE:
        return
    print(s)

def md5_hash_fn(s):
    encoded = s.encode('utf-8')
    return hashlib.md5(encoded).hexdigest()

def run_once(fn):
    has_ran = False
    output = None

    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal has_ran
        nonlocal output

        if has_ran:
            return output

        output = fn(*args, **kwargs)
        has_ran = True
        return output
    return inner


# caching function

def cache_fn(
    fn,
    path = '',
    hash_fn = md5_hash_fn,
    clear = False or CLEAR_CACHE,
    should_cache = True
):
    if not should_cache:
        return fn

    (CACHE_PATH / path).mkdir(parents = True, exist_ok = True)

    @run_once
    def clear_cache_folder_():
        cache_path = rmtree(str(CACHE_PATH / path))
        (CACHE_PATH / path).mkdir(parents = True, exist_ok = True)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        if clear:
            clear_cache_folder_()

        key = hash_fn(t)
        entry_path = CACHE_PATH / path / f'{key}.pt'

        if entry_path.exists():
            log(f'cache hit: fetching {t} from {str(entry_path)}')
            return torch.load(str(entry_path))

        out = fn(t, *args, **kwargs)

        log(f'saving: {t} to {str(entry_path)}')
        torch.save(out, str(entry_path))
        return out
    return inner
