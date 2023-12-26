import numpy as np

def array_map(f, arr: np.ndarray) -> np.ndarray:
    """
    Apply function 'f' to every deepest elements of arr and return the resulting array
    """
    res = list()
    for x in arr:
        if type(x) == np.ndarray:
            res.append(array_map(f, x))
        else:
            res.append(f(x))
    return np.array(res)

if __name__ == '__main__':
    data = [[1,5], [2,4]]
    arr = np.array(data)
    print(f'{arr}')

    doubled = array_map(lambda x: x*2, arr)
    print(f'{doubled}')
    
    tab = array_map(lambda x: x/2, doubled)
    print(f'{tab}')
