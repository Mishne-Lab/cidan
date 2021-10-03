import json
import os

import neurofinder
import pandas as pd
from scipy.io import savemat


def calc_stats(file_name, caiman, cidan, suite2p, true):
    rows = []
    caimain_file = [os.path.join(caiman, x) for x in os.listdir(caiman) if
                    file_name in x and "json" in x][0]
    cidan_file = [os.path.join(cidan, x) for x in os.listdir(cidan) if file_name in x][
        0]
    true_file = \
    [os.path.join(true, x) for x in os.listdir(true) if file_name in x and "json" in x][
        0]
    suite2p_file = [os.path.join(suite2p, x) for x in os.listdir(suite2p) if
                    file_name in x and "json" in x][0]

    with open(true_file, "r") as json_b:
        json_b_actual = json.load(json_b)
    savemat("%s.mat" % "true", {"data": json_b_actual},
            appendmat=True)
    for path, name in zip([caimain_file, cidan_file, suite2p_file],
                          ["caiman", "cidan", "suite2p"]):
        num_rois_a, num_rois_b = 0, 0
        if os.path.isfile(os.path.join(path, "roi_list.json")):
            path = os.path.join(path, "roi_list.json")

        try:
            if name == "caiman":
                with open(path, "r") as json_b:
                    json_b_actual = json.load(json_b)
                    for roi in json_b_actual:
                        roi["coordinates"] = [[x[0] + 10, x[1] + 10] for x in
                                              roi["coordinates"]]
                # savemat("caiman.mat", {"data": json_b_actual},
                #         appendmat=True).
                b = neurofinder.load(json.dumps(json_b_actual))
            else:
                # with open(path, "r") as json_b:
                #     json_b_actual = json.load(json_b)
                # savemat("%s.mat"%name, {"data": json_b_actual},
                #         appendmat=True)
                b = neurofinder.load(path)
            a = neurofinder.load(true_file)

            percision, recall = neurofinder.centers(a, b, threshold=10)
            inclusion, exclusion = neurofinder.shapes(a, b, threshold=10)
            num_rois_a, num_rois_b = a.count, b.count
            if percision != 0 and recall != 0:
                combined = 2 * percision * recall / (percision + recall)
            else:
                combined = 0
            print(path)
        except IndexError:

            percision, recall, inclusion, exclusion, combined = -2, -2, -2, -2, -2
        current_row = [name, percision, recall, inclusion, exclusion, combined,
                       num_rois_a, num_rois_b]

        rows.append(current_row)

    df = pd.DataFrame(rows, columns=["Seq", "Percision", "Recall", "Inclusion",
                                     "Exclusion", "Combined", "Num Rois truth",
                                     "Num rois detected"])
    print(df[["Seq", "Percision", "Recall", "Inclusion",
              "Exclusion", "Combined"]])

    # task_log = pandas.read_csv(args.task_log)
    # result = pandas.concat([df, task_log], axis=1)
    # df.to_csv(os.path.join(args.output_dir, f"out_{str(os.path.basename(args.output_dir[:-1]))}_{str(args.threshold)}.csv"))


if __name__ == '__main__':
    calc_stats("File5",
               "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/caiman",
               "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan",
               "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/suite2p",
               "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/true")
