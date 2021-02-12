import argparse
import json

from scipy.io import savemat

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_file", type=str, default='',
                    required=True, help="pickle input file")
parser.add_argument("-o", "--output_file", type=str, default='', required=True,
                    help="Where to output mat file",
                    )

args = parser.parse_args()
with open(args.input_file, "rb") as file:
    pickle_file = json.load(file)
    # test = {x: np.vstack(pickle_file[x]) for x in pickle_file.keys()}
    savemat(args.output_file, {"data": pickle_file},
            appendmat=True)
