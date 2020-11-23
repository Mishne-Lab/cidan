import argparse
import csv
import json
import os
from functools import reduce
from shutil import copyfile

import neurofinder
import pandas
from PIL import Image
import re

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_list", type=str, default='',
                        required=True, help="Path to all datasets in either a file or directions to a folder")
    parser.add_argument("-odir", "--output_dir", type=str, default='', required=True,
                        help="Where to output all eigen vector images and csv data",
                        )
    parser.add_argument("--roi_true", type=str, default='', required=False,
                        help="Where the roi True jsons are",
                        )
    parser.add_argument("--threshold", type=int, default=10, required=False,
                        help="Where the roi True jsons are",
                        )
    # parser.add_argument("--task_log", type=str, default='', required=True)
    args = parser.parse_args()
    if args.roi_true == '':
        roi_true_path = None
    else:
        roi_true_path = args.roi_true
    task_list = []
    if os.path.isfile(args.task_list):
        with open(args.task_list, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

            for row in reader:
                task_list.append(os.path.dirname(row[0]))
    else:
        task_list = [os.path.join(args.task_list, o) for o in os.listdir(args.task_list)
                    if os.path.isdir(os.path.join(args.task_list,o))]
        print(task_list)

    parameters_to_search = {"median_filter": [True, False], "hist_eq": [True, False],
                            "pca": [True, False], "total_num_spatial_boxes": [1],
                            "num_eig": [50], "trial_split": [True],
                            "z_score": [True,False],
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
    parameter_keys=[]
    for num, path in enumerate(task_list):
        num_rois_a, num_rois_b = 0,0
        if os.path.isfile(os.path.join(path, "roi_list.json")):

            try:
                with open(os.path.join(path, "roi_list.json"), "r") as json_b:
                    json_b_actual = json.load(json_b)
                    for roi in json_b_actual:
                        roi["coordinates"] = [[x[0]+1, x[1]+1] for x in roi["coordinates"]]


                a = neurofinder.load(os.path.join(path, "roi_true.json") if roi_true_path is None else roi_true_path)
                b = neurofinder.load(json.dumps(json_b_actual))
                percision, recall = neurofinder.centers(a, b, threshold=args.threshold)
                inclusion, exclusion = neurofinder.shapes(a, b, threshold=args.threshold)
                num_rois_a, num_rois_b = a.count, b.count
                if percision != 0 and recall != 0:
                    combined = 2 * percision *recall / ( percision + recall)
                else:
                    combined = 0
                print(path)
            except IndexError:

                percision, recall, inclusion, exclusion, combined = -2, -2, -2, -2, -2
        else:
            percision, recall, inclusion, exclusion, combined = -1, -1, -1, -1, -1
        num = int(re.search(r'\d+', path[::-1]).group()[::-1])
        current_row = [num , percision, recall, inclusion, exclusion, combined, num_rois_a, num_rois_b]
        parameter_keys = []

        with open(os.path.join(path, "parameters.json"), "r") as parameters:
            parameters = json.load(parameters)
            for x in parameters.keys():
                for y in parameters[x].keys():
                    parameter_keys.append(y)
                    current_row.append([parameters[x][y]])
        rows.append(current_row)
        files_to_copy = ["roi_outline_background.png", "roi_blob.png", "roi_blob_background.png", "pca_shape.text"]
        for file in files_to_copy:
            if os.path.isfile(os.path.join(path, file)):
                if ".png" in file:
                    Image.open(os.path.join(path,
                                            file)).save(
                        os.path.join(args.output_dir, os.path.basename(path) + "_" + file))
                else:
                    copyfile(os.path.join(path, file), os.path.join(args.output_dir, os.path.basename(path) + "_" + file))
        file = "embedding_norm_images/embedding_norm_image.png"
        if os.path.isfile(
                os.path.join(path, file)):
            Image.open(os.path.join(path,
                                    file)).save(
                os.path.join(args.output_dir, os.path.basename(path) + "_embedding_norm_image.png"))

    df = pandas.DataFrame(rows, columns=["Seq", "Percision", "Recall", "Inclusion",
                                         "Exclusion", "Combined", "Num Rois truth", "Num rois detected"] + parameter_keys)
    # task_log = pandas.read_csv(args.task_log)
    # result = pandas.concat([df, task_log], axis=1)
    df.to_csv(os.path.join(args.output_dir, f"out_{str(os.path.basename(args.output_dir[:-1]))}_{str(args.threshold)}.csv"))


if __name__ == '__main__':
    main()
