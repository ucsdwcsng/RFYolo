import os

def mkdir(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass