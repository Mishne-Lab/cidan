import os

import fire
import multiblob


def run_multiblob(data_1, data_2, data_3, data_4):
    for file in ["File1", "File2", "File3", "File4", "File5", "File6"]:
        for background in ["_median_mean", "_max", "_median_max", "_median_median",
                           "_mean", "_median", "_eigennorm"]:
            if not os.path.isdir("plots/" + file + background):
                os.mkdir("plots/" + file + background)
            folders_1 = [x for x in os.listdir(data_1) if file in x]

            if background == "_eigennorm":
                background_img = os.path.join(data_1, folders_1[0],
                                              "embedding_norm_images/embedding_norm_image.png")
            else:
                background_img = os.path.join(
                    "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/HigleyMax_Mean_Images/",
                    file + background + ".npy")
            for folder in folders_1:
                multiblob.create_graph(bg_path=background_img, shape=None, e_dir="",
                                       data_1=os.path.join(data_1, folder,
                                                           "roi_list.json"),
                                       out_file="plots/" + file + background + "/" + os.path.basename(
                                           data_1) + folder + background + "_edge.png",
                                       percent=99, blobs=False, pad=(10, 10),
                                       color_1=(255, 255, 255))
            for data in [data_3, data_4]:
                for json in [x for x in os.listdir(data) if file in x]:
                    multiblob.create_graph(bg_path=background_img, shape=None, e_dir="",
                                           data_1=os.path.join(data, json),
                                           out_file="plots/" + file + background + "/" + os.path.basename(
                                               data) + file + background + "_edge.png",
                                           pad=(10, 10),
                                           percent=99, blobs=False,
                                           color_1=(255, 255, 255))
            data = data_2
            for json in [x for x in os.listdir(data) if file in x]:
                multiblob.create_graph(bg_path=background_img, shape=None, e_dir="",
                                       data_1=os.path.join(data, json),
                                       out_file="plots/" + file + background + "/" + os.path.basename(
                                           data) + json[:-5] + background + "_edge.png",
                                       pad=(10, 10),
                                       percent=99, blobs=False,
                                       color_1=(255, 255, 255), offset=10)
            data_1_file = [x for x in os.listdir(data_1) if file in x][0]
            data_2_file = [x for x in os.listdir(data_2) if file in x][0]
            data_3_file = [x for x in os.listdir(data_3) if file in x][0]
            data_4_file = [x for x in os.listdir(data_4) if file in x][0]
            multiblob.create_graph(bg_path=background_img, shape=None, e_dir="",
                                   data_1=os.path.join(data_1, data_1_file,
                                                       "roi_list.json"),
                                   data_2=os.path.join(data_2, data_2_file),
                                   data_3=os.path.join(data_3, data_3_file),
                                   data_4=os.path.join(data_4, data_4_file),
                                   out_file="plots/" + file + background + "/" + file + background + "_all_data_edge_max.png",
                                   pad=(10, 10),
                                   percent=99, blobs=False,
                                   offset=[0, 10, 0, 0], color_1=(234, 32, 39),
                                   # red cidan
                                   color_2=(247, 159, 31)  # orange caiman
                                   , color_3=(6, 82, 221),  # blue suite2p
                                   color_4=(217, 128, 250)  # Purple true
                                   )

if __name__ == '__main__':
    fire.Fire(run_multiblob)
