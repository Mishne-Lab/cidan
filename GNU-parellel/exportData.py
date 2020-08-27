import argparse
import csv
import os
from functools import reduce

import neurofinder
import pandas
from PIL import Image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_list", type=str, default='',
                        required=True, help="Path to all datasets")
    parser.add_argument("-odir", "--output_dir", type=str, default='', required=True,
                        help="Where to output all eigen vector images and csv data",
                        )
    parser.add_argument("--task_log", type=str, default='', required=True)
    args = parser.parse_args()
    task_list = []
    with open(args.task_list, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for row in reader:
            task_list.append(row[0])

    parameters_to_search = {"median_filter": [True, False], "hist_eq": [True, False],
                            "pca": [True, False], "total_num_spatial_boxes": [1, 4],
                            "num_eig": [50], "trial_split": [True],
                            "z_score": [True, False],
                            "trial_length": [200, 400],
                            "localSpatialDenoising": [True, False]}
    total_parameters_combinations = reduce(lambda x, y: x * y,
                                           [len(parameters_to_search[x]) for x in
                                            parameters_to_search])
    print(total_parameters_combinations)
    parameter_keys = list(parameters_to_search)
    parameter_remainders = []
    for num, key in enumerate(parameter_keys):
        remainder = 1
        for x in range(num + 1):
            remainder *= len(parameters_to_search[parameter_keys[x]])
        parameter_remainders.append(remainder)

    rows = []
    for num, path in enumerate([os.path.dirname(x) for x in task_list]):
        if os.path.isfile(os.path.join(path, "roi_list.json")):
            a = neurofinder.load(os.path.join(path, "roi_true.json"))
            b = neurofinder.load(os.path.join(path, "roi_list.json"))
            percision, recall = neurofinder.centers(a, b)
            inclusion, exclusion = neurofinder.shapes(a, b)
        else:
            percision, recall, inclusion, exclusion = -1, -1, -1, -1
        current_row = [num + 1, percision, recall, inclusion, exclusion]

        for remainder, key in zip(parameter_remainders, parameter_keys):
            val = parameters_to_search[key][num % remainder // (
                    remainder // len(parameters_to_search[key]))]
            current_row.append(val)
        rows.append(current_row)
        if os.path.isfile(
                os.path.join(path, "embedding_norm_images/embedding_norm_image.png")):
            Image.open(os.path.join(path,
                                    "embedding_norm_images/embedding_norm_image.png")).save(
                os.path.join(args.output_dir, os.path.basename(path) + ".png"))

    df = pandas.DataFrame(rows, columns=["Seq", "Percision", "Recall", "Inclusion",
                                         "Exclusion"] + parameter_keys)
    task_log = pandas.read_csv(args.task_log)
    result = pandas.concat([df, task_log], axis=1)
    result.to_csv(os.path.join(args.output_dir, "out.csv"))


if __name__ == '__main__':
    main()
