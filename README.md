# CIDAN
CIDAN or Calcium Image Data Analysis is a fully featured application for preprocessing, cell detecection, and time trace analysis of 2-photon calcium imageing data of the brain. 

# Install Instructions
1. Make sure you have anaconda installed.
2. Run these commands:

```
conda create -n CIDAN python=3.7
conda activate CIDAN
pip install CIDAN
python -m CIDAN
```
If you encounter an error in the installation of hnswlib, please install Microsoft C++ build tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/ 

If you encounter another issue in the installation or running of the package please post an issue to the github repo and we will get back to you.

# Examples
We have uploaded an example dataset to google drive which is avaible here. Also you can use the data from NeuroFinder, if you want more datasets. 

# Guide to the GUI
1. First make sure your data is ready to be ingested by CIDAN. It currently only accepts tiff files. We also require the data to be pre-registered. Please make sure your data is in one of the accepted formats: 
    - A folder of tiff files representing a single trial.
    - A single tiff stack representing a single trial.
    - A folder of tiff stacks each representing a trial.
    - A folder of folders each filled with tiff files with each folder being a single trial.
2. Run CIDAN with this command:
~~~
python -m CIDAN
~~~
3. Use the open dataset tab to open the dataset
4. Next use the filter tab to apply a filter. We suggest trying at least the median filter. Z-score and histogram equalization are applied to each trial. 
5. Use the ROI Exration tab to extract the cells. Next we walk through the settings that are important in each section
    - Multiprocessing Settings: The number of spatial boxes is required to be a square number. We recommend that each spatial box is about 200x200 pixels. Each spatial box is processed in parallel and then results are combined. The spatial overlap of 60 should be fine for most cases but if you expireince many ROI's that have straight edges this number should be increased. 
    - ROI Extraction Settings: The settings for this section are mostly self explanatory. The merge temporal coefficient the lower the number the greater the difference allowed between two overlapped ROIs that are merged.
5. Next use the ROI modification sub-tab to edit each of the ROI's. To edit an ROI first use the selector brush to select pixels you want to add or subtract from the ROI. The magic wand tool uses the intermidiate saves of the cell extraction process quickly find new cells based on user input. Then select the ROI to edit by either clicking on it in the image or clicking on the checkbox next to the correct ROI. Next press the appropriate button to "Add to ROI" or "Subtract from ROI." The "New ROI from Selection button will create a new ROI at the end of the list with the current pixels that are selected. Also note that the "Delete ROI" button will renumber all the ROI's after the deleted one. 
6. Once you are satisfied with the ROIs, move to the analysis tab to view the time traces for them. CIDAN offers two main graphing modes: Many ROIs over time, and many trials over a single ROI. To select ROI's there are three methods: select them in ROI list, click on individual rois or use shift-drag to select all ROI's in a box. 
# Guide to call CIDAN in the terminal 
Warning this is significantly more complicated than using the GUI, it is recomended that you have some experience with json files and running terminal applications. 
