# cidan
cidan or Calcium Image Data Analysis is a full-feature application for preprocessing, cell detection, and time trace analysis of 2-photon calcium imaging data of the brain. 

# Install Instructions
1. Make sure you have Anaconda installed. If on windows also install Microsoft C++ build tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/ 
2. Run these commands:

```
conda create -n cidan python=3.7
conda activate cidan
pip install cidan
python -m cidan
```

If you encounter another issue in the installation or running of the package, first try rerunning 'pip install cidan'. If this doesn't fix the problem, please post the issue to the github repo and we will get back to you.

To update to the latest version of cidan, run these commands: 
```
conda activate cidan
pip install cidan --upgrade
python -m cidan
```
# Examples
We suggest using the data from NeuroFinder for example datasets. Link: http://neurofinder.codeneuro.org

# Basic Guide to the GUI
1. **First make sure your data is ready to be ingested by cidan.** It currently only accepts tiff files. We also require the data to be pre-registered. Please make sure your data is in one of the accepted formats: 
    - A folder of tiff files representing a single trial.
    - A single tiff stack representing a single trial.
    - A folder of tiff stacks, each representing a trial.
    - A folder of folders each filled with tiff files, with each folder being a single trial.
2. **Run cidan with this command:**
~~~
python -m cidan
~~~ 
**_Note the GUI will periodically freeze after you preform intensive computational actions (opening, filtering, cropping, extracting ROIs). You can view the progress of these calculations in the terminal with the provided progress bar_ <br/>**<br/>
3. **Use the *Open Dataset* tab to open the dataset.** The *Save Directory* is where we save intermediate steps and where we export all results. You can open a previous save using the *Open Previous Save* sub-tab.<br/><br/>
4. **Next use the *Preprocessing* tab to apply a filter.** We suggest trying at least the local spatial denoising and the median filter. Z-score and histogram equalization are applied to each trial individually. The filters are run in the order listed in the GUI. We also suggest applying a crop to the dataset and experimenting on a small section of the data to hone in on settings before running the algorithm on the entire dataset. For 512x512x1000 on a macbook pro applying a local spatial denoising filter, z-score, and median filter takes around 3 minutes. <br/><img src="https://github.com/Mishne-Lab/cidan/blob/master/images/Preprocessing.png" width="400">

5. **Use the *ROI Extraction* tab to extract the cells.** This should take around 5 minutes with the default settings on a 512x512x1000 dataset. Next we walk through the settings that are important in each section:
    - **Multiprocessing Settings:** The number of spatial boxes is required to be a square number. We recommend that each spatial box is about 200x200 pixels. Each spatial box is processed in parallel and then results are combined. The spatial overlap of 60 should be fine for most cases but if you experience many ROI's that have straight edges this number should be increased. 
    - **ROI Extraction Settings:** The settings for this section are mostly self explanatory. The “merge temporal coefficient” lets you control if two overlapping ROIs are merged based on the Pearson correlation of their time traces. The lower the number the greater the difference. 
    - **Advanced Settings:** There are two important settings here: Eigen vector number and number of timesteps. The default amount of eigen vectors will usually work but if the result isn't detecting all the ROIs, we recommend increasing this number to 75 or 100. If when you look at the eigen norm image, present in the ROI Display Settings, the image doesn't express the ROI's well then increase the Eigen Accuracy to 7 or 8.<br/> <img src="https://github.com/Mishne-Lab/cidan/blob/master/images/ROI%20Extraction.png" width="400">

5. **Next use the *ROI modification* sub-tab to edit each of the ROI's.** To edit an ROI, first use the selector brush to select pixels you
want to add or subtract from the ROI. The magic wand tool uses the intermediate saves of the cell extraction process to quickly find new cells based on user input. Then select the ROI to edit by either clicking on it in the image or clicking on the checkbox next to the correct ROI. Next, press the appropriate button to "Add to Selected ROI" or "Subtract from Selected ROI." The "Create ROI from Selection” button will create a new ROI at the end of the list with the current pixels that are selected. Also note that the "Delete ROI" button will renumber all the ROI's after the deleted one. You can also use keyboard shortcuts for these operations, they are notated on the GUI. 
6. **Once you are satisfied with the ROIs, move to the analysis tab to view the time traces for them.** cidan offers two main graphing modes: Many ROIs over time, and many trials over a single ROI. There are three methods to select ROIs: select them in the ROI list, click on individual rois, or use shift-drag to select all ROI's in a box.<br/> <img src="https://github.com/Mishne-Lab/cidan/blob/master/images/Analysis.png" width="400">
7. **To export the results, use the top bar and select export.** If you have trouble selecting the top bar, there is small known bug, you can fix it by just selecting another window then going back and selecting cidan again. 
# Improving your results
1. **Look at the embedding norm image** (found in the display settings tab). 
    - **If the embedding norm image localizes in on most but not all of your ROIs,** we suggest increasing the number of eigen vectors generated to 75 or 100. 
    - **If the embedding norm image has all the ROIs,** then we suggest first decreasing the ROI circuity threshold to 0, and then if that doesn't work increasing the number of iterations to 200 or 300 could also help. Another possible fix if you only have less than 6 trials is that we aren't selecting enough eigen vectors, to fix this increase Eigen threshold value to .3 or .5. 
    - **If the eigen norm image has wierd lines or other artifacts**, we suggest increasing the smoothing filters that are applied in the preprocessing step. Also it could help to increase the Eigen Accuracy parameter to 7 or 8. 
# Guide to call cidan in the terminal 
Warning this is significantly more complicated than using the GUI. It is recommended that you have some experience with json files and running terminal applications. 
1. Download the default parameter.json file from the github repo. Place it in the folder you want to save the results in. The other option is to use the parameter.json file from a save directory of another similar dataset. This will use all the settings from that dataset.
2. Next use a program of your choice like text-edit to edit the parameter.json file. Here are the settings that you should look into changing. 
    - "dataset_folder_path": change to the folder containing the dataset you want to process. 
    - "trials_loaded": this is a list of all file or folder names in said folder that you want to process
    - "trials_all": this should be the same as "trials_loaded"
    - "total_num_time_steps": only applicable if you are processing a single trial, if you are set this parameter to ceil(number of timesteps/250)
    - "total_num_spatial_boxes": Set so each spatial box is approximately 200x200. Note the number must be a square.
3. Now use terminal to run this command 
~~~
python -m cidan -headless True --parameter <path to parameter.json file> 
~~~
4. This will export everything into the directory of the parameter file; see the export section to understand each of the export files. To easily view the results, you can use the open previous save dir option in the GUI to open this directory.
# Export Files
This is an explanation of all the files exported by cidan into the specified save directory. 
- parameters.json: file containing all the settings for the algorithm
- time_traces_DeltaF_Over_F.pickle: A file with a list of the deltaF/F for each roi in order. In this format: [  [  [time trace for ROI#1 trial 1],[time trace for ROI#1 trial 2]],  [[time trace for ROI#2 trial 1],[time trace for ROI#2 trial 2]  ] ]. Each time trace is stored as numpy ndarray. 
- time_traces_Mean.pickle: Same format as previous file except for using the mean fluorescence of ROI at each time point for time trace. 
- rois.pickle: Internal use
- eigen_vectors folder: Internal use
- embedding_norm_images: Internal use


