
# Adapted from https://medium.com/@nkhaja/memoization-and-decorators-with-python-32f607439f84
def memoize(func):
    
    cache = {}    
    
    def memoized_func(*args, **kwargs):
        
        key = str(args) + str(kwargs)
        
        if key not in cache:
            
            cache[key] = func(*args, **kwargs)
            
        return cache[key]    
    
    return memoized_func
    
