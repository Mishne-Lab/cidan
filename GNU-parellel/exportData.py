import argparse
import csv
import json
import os

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

    parameter_keys = []

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
        parameter_keys = []

        with open(os.path.join(path, "parameters.json"), "r") as parameters:
            parameters = json.load(parameters)
            for x in parameters.keys():
                for y in parameters[x].keys():
                    parameter_keys.append(y)
                    current_row.append([parameters[x][y]])


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
