# merge two dictionaries
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# check if x is int
def is_int(x):
    try:
        return int(x)==x
    except:
        return False


# check if the response contains only integers
def is_factor(y):
    return all(is_int(item) for item in y)


# flatten list of lists
flatten = lambda l: [
    item for sublist in l for item in sublist
]
