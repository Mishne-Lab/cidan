import argparse
import csv
import json
import os
from functools import reduce
from shutil import copyfile
import numpy as np
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

    def save_image(image, path):

        img = Image.fromarray(
            (np.reshape(image - image.min(),
                        (300, 300, 3)) / (
                     image.max() - image.min()) * 255).astype("uint8"))
        img.save(path)

    size = 256
    image = np.zeros((size, size, 3), dtype="int")
    color_list = [(218, 67, 34),
                  (132, 249, 22), (22, 249, 140), (22, 245, 249),
                  (22, 132, 249), (224, 22, 249), (249, 22, 160)]
    color_list_len = len(color_list)
    with open(roi_true_path, "r") as json_true:
        json_b_actual = json.load(json_true)
    for num, x in enumerate(json_b_actual):
        cords = x["coordinates"]
        for pixel in cords:
            image[pixel[0]-1, pixel[1]-1] += color_list[num % color_list_len]
    image_true=image[10:245,10:245,:]
    # parameters_to_search = {"median_filter": [True, False], "hist_eq": [True, False],
    #                         "pca": [True, False], "total_num_spatial_boxes": [1],
    #                         "num_eig": [50], "trial_split": [True],
    #                         "z_score": [True,False],
    #                         "trial_length": [200, 400],
    #                         "localSpatialDenoising": [True, False]}
    # total_parameters_combinations = reduce(lambda x, y: x * y,
    #                                        [len(parameters_to_search[x]) for x in
    #                                         parameters_to_search])
    # print(total_parameters_combinations)
    # parameter_keys = list(parameters_to_search)
    # parameter_remainders = []
    # for num, key in enumerate(parameter_keys):
    #     remainder = 1
    #     for x in range(num + 1):
    #         remainder *= len(parameters_to_search[parameter_keys[x]])
    #     parameter_remainders.append(remainder)

    rows = []
    parameter_keys=[]
    for num, path in enumerate(task_list):
        num_rois_a, num_rois_b = 0,0
        if os.path.isfile(os.path.join(path, "roi_list.json")) and os.path.basename(args.roi_true)[:5] in path:

            try:
                with open(os.path.join(path, "roi_list.json"), "r") as json_b:
                    json_b_actual = json.load(json_b)
                    for roi in json_b_actual:
                        roi["coordinates"] = [[x[0]+11, x[1]+11] for x in roi["coordinates"]]


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
            if 'min_neuropil_pixels' not in parameters["time_trace_params"].keys():
                parameters["time_trace_params"]['min_neuropil_pixels'] = 0
            if 'time_trace_type' not in parameters["time_trace_params"].keys():
                parameters["time_trace_params"]['time_trace_type'] = 'Mean'
            if 'denoise' not in parameters["time_trace_params"].keys():
                parameters["time_trace_params"]['denoise'] = True
            if "roi_eccentricity_limit" not in parameters["roi_extraction_params"].keys():
                parameters["roi_extraction_params"]["roi_eccentricity_limit"] = 1
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
                    if file=="roi_blob.png":
                        try:
                            roi_blob = np.array(Image.open(os.path.join(path,
                                                    file)))
                            embed = np.array(Image.open(os.path.join(path,
                                        "embedding_norm_images/embedding_norm_image.png")))

                            roi_image_blob_w_background = roi_blob + (np.dstack(
                [embed, embed, embed])  / embed.max() * 255)
                            roi_image_blob_w_background = (roi_image_blob_w_background/roi_image_blob_w_background.max()*255).astype(np.uint8)
                            roi_image_pil = Image.fromarray(roi_image_blob_w_background)
                            roi_image_pil.save(
                    os.path.join(args.output_dir, os.path.basename(path) + "_embedding_norm_image_blob.png"))


                            roi_image_true_blob_w_background = image_true + (np.dstack(
                                [embed, embed, embed]) / embed.max() * 255)
                            roi_image_true_blob_w_background = (
                                        roi_image_true_blob_w_background / roi_image_true_blob_w_background.max() * 255).astype(
                                np.uint8)
                            roi_image_pil = Image.fromarray(roi_image_true_blob_w_background)
                            roi_image_pil.save(
                                os.path.join(args.output_dir,
                                             os.path.basename(path) + "_embedding_norm_image_blob_true.png"))
                        except ValueError:
                            print("Error on number: "+os.path.basename(path))

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
