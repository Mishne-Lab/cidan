import argparse
import pickle

import numpy as np
from scipy.io import savemat

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_file", type=str, default='',
                    required=True, help="pickle input file")
parser.add_argument("-o", "--output_dir", type=str, default='', required=True,
                    help="Where to output mat file",
                    )

args = parser.parse_args()
with open(args.input_file, "rb") as file:
    pickle_file = pickle.load(file)
    test = {x[:31].replace(" ", "_"): np.vstack(pickle_file[x]) for x in
            pickle_file.keys()}
    savemat(args.output_dir + "test" + ".mat", {"data": test},
            appendmat=True)
    for x in pickle_file.keys():
        savemat(args.output_dir + x + ".mat", {"data": np.vstack(pickle_file[x])},
                appendmat=True)
