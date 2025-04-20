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
    """
    Determine if the target variable `y` is for classification (True) or regression (False).
    
    Parameters:
    y : array-like
        Target variable (labels/response variable)
    
    Returns:
    bool
        True if `y` is categorical (classification), False if numeric (regression)
    """
    y_array = np.asarray(y)
    
    # Boolean → classification
    if y_array.dtype == bool:
        return True
    
    # Strings or objects → classification
    if y_array.dtype.kind in ['U', 'S', 'O']:
        return True
    
    # Numeric types (int/float) with few unique values → classification
    if y_array.dtype.kind in ['i', 'u', 'f']:
        unique_values = np.unique(y_array[~np.isnan(y_array)])  # Exclude NaNs
        if len(unique_values) <= 10:  # Threshold for classification
            return True
    
    # Otherwise → regression
    return False

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
