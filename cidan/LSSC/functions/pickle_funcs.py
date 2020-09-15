import os
import pickle

dir_name = "/data2/Sam/pythonTestEnviroment/tmp_pickle"


def pickle_set_dir(dir_path):
    global dir_name
    dir_name = dir_path


def pickle_save(obj, name, trial_num=0, output_directory=None):
    """
    saves an object as a pickle file in a folder specific to the trial
    Parameters
    ----------
    obj: object to pickle
    name: name of file
    trial_num: which trial this is part of
    output_directory: optional if want to save in specific directory
    Returns
    -------
    None
    """

    if ".pickle" not in name:
        name = name + ".pickle"

    name = name.replace(" ", "_")

    if output_directory:
        dir_full_path = output_directory
    else:
        dir_full_path = os.path.join(dir_name, str(trial_num) + "/")
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
    if not os.path.isdir(dir_full_path):
        os.mkdir(dir_full_path)
    with open(os.path.join(dir_full_path, name), "wb") as file:
        pickle.dump(obj, file, protocol=4)


def pickle_exist(name, trial_num=0, output_directory=None):
    """
    checks if a pickle file exists
    Parameters
    ----------
    name: name of file
    trial_num: which trial this is part of

    Returns
    -------
    None
    """
    name = name.replace(" ", "_")
    if not output_directory:
        dir_full_path = os.path.join(dir_name, str(trial_num) + "/")
    else:
        dir_full_path = output_directory
    if ".pickle" not in name:
        name = name + ".pickle"
    return os.path.isfile(os.path.join(dir_full_path, name))


def pickle_load(name, trial_num=0, output_directory=None):
    """
    loads a pickled object
    Parameters
    ----------
    name name of file
    trial_num which trial this is part of

    Returns
    -------
    object
    """
    name = name.replace(" ", "_")
    if not output_directory:
        dir_full_path = os.path.join(dir_name, str(trial_num) + "/")
    else:
        dir_full_path = output_directory
    if ".pickle" not in name:
        name = name + ".pickle"
    with open(os.path.join(dir_full_path, name), "rb") as file:
        return pickle.load(file)
    raise FileNotFoundError


def pickle_clear(trial_num=0):
    """
    Clears all pickle files saved in temp
    Parameters
    ----------
    trial_num which trial this is part of
    Returns
    -------
    None
    """
    try:
        dir_full_path = os.path.join(dir_name, str(trial_num) + "/")
        filelist = [f for f in os.listdir(dir_full_path) if
                    f.endswith(".pickle")]
        for f in filelist:
            os.remove(os.path.join(dir_full_path, f))
    except FileNotFoundError:
        pass
