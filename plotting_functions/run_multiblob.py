import os

import fire
import multiblob


def run_multiblob(data_1, data_2, data_3, data_4):
    for file in ["File1", "File2", "File3", "File4", "File5", "File6"]:
        folders_1 = [x for x in os.listdir(data_1) if file in x]
        background_img = os.path.join(data_1, folders_1[0],
                                      "embedding_norm_images/embedding_norm_image.png")
        for folder in folders_1:
            multiblob.create_graph(bg_path=background_img, shape=None, e_dir="",
                                   data_1=os.path.join(data_1, folder, "roi_list.json"),
                                   out_file="plots/" + os.path.basename(
                                       data_1) + folder + "_edge.png",
                                   percent=99, blobs=False,
                                   color_1=(255, 255, 255))
        for data in [data_2, data_3, data_4]:
            for json in [x for x in os.listdir(data) if file in x]:
                multiblob.create_graph(bg_path=background_img, shape=None, e_dir="",
                                       data_1=os.path.join(data, json),
                                       out_file="plots/" + os.path.basename(
                                           data) + file + "_edge.png",
                                       percent=99, blobs=False,
                                       color_1=(255, 255, 255))


if __name__ == '__main__':
    fire.Fire(run_multiblob)
