import numpy as np
import pandas as pd


def dict_to_dataframe_series(data, series_names):
    df = pd.DataFrame(
        np.zeros((len(data["Model"]), 2)), columns=["Model", "Time Taken"]
    )
    for key, value in data.items():
        if all(hasattr(elt, "__len__") for elt in value) and key not in (
            "Model",
            "Time Taken",
        ):
            for i, elt1 in enumerate(value):
                for j, elt2 in enumerate(elt1):
                    df.loc[i, f"{key}_{series_names[j]}"] = elt2
        else:
            df[key] = value
    return df


# flatten list of lists
# flatten = lambda l: [item for sublist in l for item in sublist]
def flatten(x):
    res = []

    for elt1 in x:
        try:
            for elt2 in elt1:
                res.append(elt2)

        except:
            res.append(elt1)

    return res


# check if the response contains only integers
def is_factor(y):
    # return not np.mod(y, 1).any()    
    return np.issubdtype(y.dtype, np.integer)


# check if x is float
def is_float(x):
    return isinstance(x, float)


# check if x is int
def is_int(x):
    try:
        return int(x) == x
    except:
        return False


# merge two dictionaries
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# apply function to a tuple (element-wise)
def tuple_map(x, foo):
    return tuple(map(foo, x))
