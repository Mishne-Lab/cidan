import argparse
import csv
import json
import os
from functools import reduce


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-idir", "--input_dir", type=str, default='',
                        required=True, help="Path to all datasets")
    parser.add_argument("-odir", "--output_dir", type=str, default='', required=True,
                        help="Where to output all data",
                        )
    parser.add_argument("-p", "--parameter", type=str, default='', required=True,
                        help="Parameter file to base off of",
                        )
    args = parser.parse_args()
    with open(args.parameter, "r") as f:
        parameter_json = json.load(f)
    parameters_to_search = {"median_filter": [True, False], "hist_eq": [True, False],
                            "pca": [True, False], "total_num_spatial_boxes": [1, 4, 9],
                            "num_eig": [50], "trial_split": [True],
                            "trial_length": [250, 500, 1000],
                            "localSpatialDenoising": [True]}
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

    directory_list = list_dirs(args.input_dir, 1)

    with open("task_list.csv", mode="w") as task_list:
        task_list_writer = csv.writer(task_list, delimiter=',', quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)

        for curr_dir in directory_list:

            for x in range(total_parameters_combinations):
                curr_json = parameter_json.copy()
                curr_out_dir = os.path.join(args.output_dir, curr_dir + str(x))
                if not os.path.isdir(curr_out_dir):
                    os.mkdir(curr_out_dir)
                for remainder, key in zip(parameter_remainders, parameter_keys):
                    val = parameters_to_search[key][x % remainder // (
                            remainder // len(parameters_to_search[key]))]
                    for section in curr_json:
                        if key in curr_json[section].keys():
                            curr_json[section][key] = val
                curr_json["dataset_params"]["dataset_folder_path"] = args.input_dir
                curr_json["dataset_params"][
                    "original_folder_trial_split"] = os.path.basename(curr_dir)
                if os.path.isdir(os.path.join(args.input_dir, curr_dir, "images")):
                    curr_json["dataset_params"]["dataset_folder_path"] = os.path.join(
                        args.input_dir, curr_dir)
                    curr_json["dataset_params"][
                        "original_folder_trial_split"] = "images"
                    with open(os.path.join(args.input_dir, curr_dir,
                                           "regions/regions.json"), "r") as f:
                        with open(os.path.join(curr_out_dir, "roi_true.json"),
                                  "w") as f2:
                            json.dump(json.load(f), f2)

                parameter_file_path = os.path.join(curr_out_dir, "parameter.json")
                with open(parameter_file_path, "w") as f:
                    json.dump(curr_json, f)

                task_list_writer.writerow([parameter_file_path])


def list_dirs(dir, depth_left):
    if depth_left == 1:
        return [x.name for x in os.scandir(dir)]
    else:
        dir_full_list = []
        for dir_to_search in [x.name for x in os.scandir(dir)]:
            dir_full_list += list_dirs(dir_to_search, depth_left)
        return dir_full_list


if __name__ == '__main__':
    main()
