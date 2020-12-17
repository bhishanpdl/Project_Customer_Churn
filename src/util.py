import numpy as np
import pandas as pd

import sklearn
import config

def show_methods(obj, ncols=7,start=None, inside=None):
    """ Show all the attributes of a given method.
    Example:
    ========
    show_method_attributes(list)
    """

    print(f'Object Type: {type(obj)}\n')
    lst = [elem for elem in dir(obj) if elem[0]!='_' ]
    lst = [elem for elem in lst
            if elem not in 'os np pd sys time psycopg2'.split() ]

    if isinstance(start,str):
        lst = [elem for elem in lst if elem.startswith(start)]

    if isinstance(start,tuple) or isinstance(start,list):
        lst = [elem for elem in lst for start_elem in start
                if elem.startswith(start_elem)]

    if isinstance(inside,str):
        lst = [elem for elem in lst if inside in elem]

    if isinstance(inside,tuple) or isinstance(inside,list):
        lst = [elem for elem in lst for inside_elem in inside
                if inside_elem in elem]

    return pd.DataFrame(np.array_split(lst,ncols)).T.fillna('')

def print_time_taken(time_taken):
    h,m = divmod(time_taken,60*60)
    m,s = divmod(m,60)
    time_taken = f"{h:.0f} h {m:.0f} min {s:.2f} sec" if h > 0 else f"{m:.0f} min {s:.2f} sec"
    time_taken = f"{m:.0f} min {s:.2f} sec" if m > 0 else f"{s:.2f} sec"

    print(f"\nTime Taken: {time_taken}")
