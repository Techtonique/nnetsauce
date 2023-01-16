import numpy as np

# merge two dictionaries
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# check if x is int
def is_int(x):
    try:
        return int(x) == x
    except:
        return False


# check if x is float
def is_float(x):
    return isinstance(x, float)


# check if the response contains only integers
def is_factor(y):
    #return not np.mod(y, 1).any()
    return not np.array([is_float(y[i]) for i in range(len(y))]).all()


# flatten list of lists
#flatten = lambda l: [item for sublist in l for item in sublist]
def flatten(x): 
            
    res = []    
    
    for elt1 in x:
                    
        try:
            
            for elt2 in elt1:
                
                res.append(elt2)  
                
        except:
            
            res.append(elt1)  
                                
    return res

