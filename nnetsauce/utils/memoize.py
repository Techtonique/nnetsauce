import functools


def memoize(func, maxsize=128):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)

        if key in cache:
            return cache[key]

        # else: key not in cache
        # you don't want it to grow indefinitely
        if len(cache) > maxsize:
            cache.clear()

        cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_func
