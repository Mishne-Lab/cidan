#/bin/bash
rm task_list.csv
python createTaskList.py -idir ~/Desktop/HigleyData/roi80 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 80 --num_spatial_boxes 1 --start 200
python createTaskList.py -idir ~/Desktop/HigleyData/roi80 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 70 --num_spatial_boxes 1 --start 224
python createTaskList.py -idir ~/Desktop/HigleyData/roi40 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 35 --num_spatial_boxes 1 --start 224
python createTaskList.py -idir ~/Desktop/HigleyData/roi40 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 45 --num_spatial_boxes 1 --start 200
python createTaskList.py -idir ~/Desktop/HigleyData/roi160 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 160 --num_spatial_boxes 1 --start 200
python createTaskList.py -idir ~/Desktop/HigleyData/roi160 -odir ~/Desktop/HigleyDataOut5 -p ~/Desktop/CIDAN/GNU-parellel/parameters.json --num_rois 130 --num_spatial_boxes 1 --start 224

