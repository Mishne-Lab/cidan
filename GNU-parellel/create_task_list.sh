#/bin/bash
rm task_list.csv
python createTaskList.py -idir ~/Desktop/HigleyData/roi80 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 80 --num_spatial_boxes 1 --start 100
#python createTaskList.py -idir ~/Desktop/HigleyData/roi80 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 20 --num_spatial_boxes 4 --start 36
#python createTaskList.py -idir ~/Desktop/HigleyData/roi40 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 15 --num_spatial_boxes 4 --start 36
python createTaskList.py -idir ~/Desktop/HigleyData/roi40 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 45 --num_spatial_boxes 1 --start 100
python createTaskList.py -idir ~/Desktop/HigleyData/roi160 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 160 --num_spatial_boxes 1 --start 100
#python createTaskList.py -idir ~/Desktop/HigleyData/roi160 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 40 --num_spatial_boxes 4 --start 36

