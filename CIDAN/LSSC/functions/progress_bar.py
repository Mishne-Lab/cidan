# Print iterations progress
import os


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100,
                     fill='█', printEnd="\r", progress_signal=None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    # print("test")
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    if progress_signal is not None:

        progress_signal.sig.emit(prefix, 100 * (iteration / float(total)))
    else:
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)

        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()


def printProgressBarROI(total_num_spatial_boxes, total_num_time_steps, save_dir,
                        iteration_last_overide=-1, progress_signal=None):
    total = total_num_time_steps * total_num_spatial_boxes * 2 + total_num_spatial_boxes + 3
    if iteration_last_overide != -1:
        printProgressBar(total - iteration_last_overide,
                         total=total,
                         prefix="ROI Extraction Progress:",
                         suffix="Complete")
    filelist = [f for f in os.listdir(os.path.join(save_dir, "eigen_vectors"))]
    filelist2 = [f for f in os.listdir(os.path.join(save_dir, "temp_files/embedding"))]
    filelist3 = [f for f in os.listdir(os.path.join(save_dir, "temp_files/rois"))]

    printProgressBar(len(filelist) + len(filelist2) + len(filelist3),
                     total=total_num_time_steps * total_num_spatial_boxes * 2 + total_num_spatial_boxes + 3,
                     prefix="ROI Extraction Progress:",
                     suffix="Complete", progress_signal=progress_signal)


def printProgressBarFilter(total_num_spatial_boxes, total_num_time_steps, save_dir,
                           progress_signal=None):
    total = total_num_time_steps * total_num_spatial_boxes * 2 + total_num_spatial_boxes + 3
    filelist2 = [f for f in os.listdir(os.path.join(save_dir, "temp_files/filter"))]

    printProgressBar(len(filelist2), total=total_num_time_steps + 2,
                     prefix="Preprocessing Progress:",
                     suffix="Complete", progress_signal=progress_signal)
