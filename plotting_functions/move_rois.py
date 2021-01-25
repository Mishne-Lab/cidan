import json

import fire


def move_rois(file, out_file, amount=(0, 0)):
    with open(file, "r") as json_b:
        json_b_actual = json.load(json_b)
        for roi in json_b_actual:
            roi["coordinates"] = [[x[0] + amount[0], x[1] + amount[1]] for x in
                                  roi["coordinates"]]
        with open(out_file, "w") as out:
            json.dump(json_b_actual, out)


if __name__ == '__main__':
    fire.Fire(move_rois)
