#/bin/bash
rm task_list.csv
python createTaskList.py -idir ~/Desktop/HigleyData/roi80 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 80 --num_spatial_boxes 1 --start 300
python createTaskList.py -idir ~/Desktop/HigleyData/roi80 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 70 --num_spatial_boxes 1 --start 372
python createTaskList.py -idir ~/Desktop/HigleyData/roi40 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 35 --num_spatial_boxes 1 --start 372
python createTaskList.py -idir ~/Desktop/HigleyData/roi40 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 30 --num_spatial_boxes 1 --start 300
python createTaskList.py -idir ~/Desktop/HigleyData/roi160 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 160 --num_spatial_boxes 1 --start 300
python createTaskList.py -idir ~/Desktop/HigleyData/roi160 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 130 --num_spatial_boxes 1 --start 372

