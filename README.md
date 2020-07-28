# CIDAN
CIDAN or Calcium Image Data Analysis is a full-feature application for preprocessing, cell detection, and time trace analysis of 2-photon calcium imaging data of the brain. 

# Install Instructions
1. Make sure you have Anaconda installed.
2. Run these commands:

```
conda create -n CIDAN python=3.7
conda activate CIDAN
pip install CIDAN
python -m CIDAN
```
If you encounter an error in the installation of hnswlib, please install Microsoft C++ build tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/ 

If you encounter another issue in the installation or running of the package, please post the issue to the github repo and we will get back to you.

# Examples
We have uploaded an example dataset to google drive which is available here. Also you can use the data from NeuroFinder, if you want more datasets. 

# Guide to the GUI
1. First make sure your data is ready to be ingested by CIDAN. It currently only accepts tiff files. We also require the data to be pre-registered. Please make sure your data is in one of the accepted formats: 
    - A folder of tiff files representing a single trial.
    - A single tiff stack representing a single trial.
    - A folder of tiff stacks, each representing a trial.
    - A folder of folders each filled with tiff files, with each folder being a single trial.
2. Run CIDAN with this command:
~~~
python -m CIDAN
~~~
3. **Use the *Open Dataset* tab to open the dataset.** The *Save Directory* is where we save intermediate steps and where we export all results. You can open a previous save using the *Open Previous Save* sub-tab.
4. **Next use the *Preprocessing* tab to apply a filter.** We suggest trying at least the median filter. Z-score and histogram equalization are applied to each trial. We also suggest applying a crop to the dataset and experimenting on a small section of the data to hone in on settings before running the algorithm on the entire dataset. 
5. **Use the *ROI Extraction* tab to extract the cells.** Next we walk through the settings that are important in each section:
    - **Multiprocessing Settings:** The number of spatial boxes is required to be a square number. We recommend that each spatial box is about 200x200 pixels. Each spatial box is processed in parallel and then results are combined. The spatial overlap of 60 should be fine for most cases but if you experience many ROI's that have straight edges this number should be increased. 
    - **ROI Extraction Settings:** The settings for this section are mostly self explanatory. The “merge temporal coefficient” lets you control if two overlapping ROIs are merged based on the Pearson correlation of their time traces. The lower the number the greater the difference. 
    - **Advanced Settings:** There are two important settings here: Eigen vector number and number of timesteps. The default amount of eigen vectors will usually work but if the result isn't detecting all the ROIs, we recommend increasing this number to 75 or 100. 
5. **Next use the *ROI modification* sub-tab to edit each of the ROI's.** To edit an ROI, first use the selector brush to select pixels you want to add or subtract from the ROI. The magic wand tool uses the intermediate saves of the cell extraction process to quickly find new cells based on user input. Then select the ROI to edit by either clicking on it in the image or clicking on the checkbox next to the correct ROI. Next, press the appropriate button to "Add to ROI" or "Subtract from ROI." The "New ROI from Selection” button will create a new ROI at the end of the list with the current pixels that are selected. Also note that the "Delete ROI" button will renumber all the ROI's after the deleted one. 
6. Once you are satisfied with the ROIs, move to the analysis tab to view the time traces for them. CIDAN offers two main graphing modes: Many ROIs over time, and many trials over a single ROI. There are three methods to select ROIs: select them in the ROI list, click on individual rois, or use shift-drag to select all ROI's in a box. 
7. **To export the results, use the top bar and select export.** If you have trouble selecting the top bar, there is small known bug, you can fix it by just selecting another window then going back and selecting CIDAN again. 
# Guide to call CIDAN in the terminal 
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
python -m CIDAN -headless True --parameter <path to parameter.json file> 
~~~
4. This will export everything into the directory of the parameter file; see the export section to understand each of the export files. To easily view the results, you can use the open previous save dir option in the GUI to open this directory.
# Export Files
This is an explanation of all the files exported by CIDAN into the specified save directory. 
- parameters.json: file containing all the settings for the algorithm
- time_traces_DeltaF_Over_F.pickle: A file with a list of the deltaF/F for each roi in order. In this format: [  [  [time trace for ROI#1 trial 1],[time trace for ROI#1 trial 2]],  [[time trace for ROI#2 trial 1],[time trace for ROI#2 trial 2]  ] ]. Each time trace is stored as numpy ndarray. 
- time_traces_Mean.pickle: Same format as previous file except for using the mean fluorescence of ROI at each time point for time trace. 
- rois.pickle: Internal use
- eigen_vectors folder: Internal use
- embedding_norm_images: Internal use


