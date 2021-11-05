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
    parser.add_argument("--num_rois", type=int, default=70, required=False,
                        help="Number of rois",
                        )
    parser.add_argument("--num_spatial_boxes", type=int, default=1, required=False,
                        help="Number of rois",
                        )
    parser.add_argument("--start",type=int, default=0, required=False, )
    args = parser.parse_args()
    with open(args.parameter, "r") as f:
        parameter_json = json.load(f)
    parameters_to_search = {"median_filter": [True], "hist_eq": [False,True],"z_score": [True],
                            "pca": [False], "total_num_spatial_boxes": [1],
                            "num_eig": [51], "trial_split": [True],
                            "trial_length": [500],"eigen_accuracy":[10],
                            "localSpatialDenoising": [False,True], "auto_crop":[False],
                            #"num_rois":[175],
                            "eigen_threshold_value":[0], "roi_eccentricity_limit": [1],
                            "elbow_threshold_value": [.95],"max_iter":[300], "merge_threshold":[1],
        "roi_circ_threshold":[0],
                            # "roi_eccentricity_limit": [1],
                            # "crop_x": [[
                            #     0,
                            #     256
                            # ]],
                            # "crop_y": [[
                            #     0,
                            #     256
                            # ]]
                            }
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
    if os.path.isdir(args.input_dir):
        directory_list = list_dirs(args.input_dir, 1)
        directory_list = [x for x in directory_list if os.path.isfile(os.path.join(args.input_dir, x))]
    else:
        directory_list = [os.path.basename(args.input_dir)]

    with open("task_list.csv", mode="a") as task_list:
        task_list_writer = csv.writer(task_list, delimiter=',', quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)

        for curr_dir in directory_list:

            for x in range(total_parameters_combinations):
                for y in range(1):
                    curr_json = parameter_json.copy()
                    curr_out_dir = os.path.join(args.output_dir, curr_dir + str(x+args.start))
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
                        "original_folder_trial_split"] = [os.path.basename(curr_dir)]
                    curr_json["dataset_params"][
                        "trials_all"] = [os.path.basename(curr_dir)]
                    curr_json["dataset_params"][
                        "trials_loaded"] = [os.path.basename(curr_dir)]
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

                    parameter_file_path = os.path.join(curr_out_dir, "parameters.json")
                    with open(parameter_file_path, "w") as f:
                        json.dump(curr_json, f)

                    task_list_writer.writerow([curr_out_dir])
                    print(curr_out_dir)


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
