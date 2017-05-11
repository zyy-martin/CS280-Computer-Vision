import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


def load_data(output_path, img_dir, label_txt = None):
    if label_txt is None:
        filenames = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
        y = None
    else:
        files = open(label_txt, 'r')
        lines = files.readlines()
        filenames = [line.split(' ')[0] for line in lines]
        y = [int(line.split(' ')[1].split('\n')[0]) for line in lines]
    X = {filename:plt.imread(filename) for filename in filenames}
    obj = {'X':X, 'filenames':filenames, 'y':y}
    np.save(output_path, obj)
    return 0
