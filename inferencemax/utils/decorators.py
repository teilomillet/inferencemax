# inferencemax/utils/decorators.py

import time

from functools import wraps

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed_time:.2f} seconds.")
        return result
    return wrapper