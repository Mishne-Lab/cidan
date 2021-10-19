#/bin/bash
rm task_list.csv
python createTaskList.py -idir /data/HigleyData/file1/ -odir /data/HigleyDataMishneClass -p ~/Desktop/CIDAN/GNU-parellel/parameters_file1.json --num_spatial_boxes 1 --start 10
python createTaskList.py -idir /data/HigleyData/file2 -odir /data/HigleyDataMishneClass -p ~/Desktop//CIDAN/GNU-parellel/parameters_file2.json --start 10
python createTaskList.py -idir /data/HigleyData/file4 -odir /data/HigleyDataMishneClass -p ~/Desktop/CIDAN/GNU-parellel/parameters_file4.json --start 10
python createTaskList.py -idir /data/HigleyData/file5 -odir /data/HigleyDataMishneClass -p ~/Desktop/CIDAN/GNU-parellel/parameters_file5.json --start 10
python createTaskList.py -idir /data/HigleyData/file3 -odir /data/HigleyDataMishneClass -p ~/Desktop/CIDAN/GNU-parellel/parameters_file3.json --start 10
python createTaskList.py -idir /data/HigleyData/file6 -odir /data/HigleyDataMishneClass -p ~/Desktop/CIDAN/GNU-parellel/parameters_file6.json --start 10
python createTaskList.py -idir /data/Neurofinder_tifs/ -odir /data/NeurofinderClassOutputs -p ~/Desktop/CIDAN/GNU-parellel/parameters_file6.json --start 10


#python createTaskList.py -idir /data/HigleyData/roi40 -odir /data/HigleyDataOut6 -p /data/CIDAN/GNU-parellel/parameters.json --num_rois 35 --num_spatial_boxes 1 --start 372
#python createTaskList.py -idir /data/HigleyData/roi40 -odir /data/HigleyDataOut6 -p /data/CIDAN/GNU-parellel/parameters.json --num_rois 30 --num_spatial_boxes 1 --start 300
#python createTaskList.py -idir /data/HigleyData/roi160 -odir /data/HigleyDataOut6 -p /data/CIDAN/GNU-parellel/parameters.json --num_rois 160 --num_spatial_boxes 1 --start 300
#python createTaskList.py -idir /data/HigleyData/roi160 -odir /data/HigleyDataOut6/ -p /data/CIDAN/GNU-parellel/parameters.json --num_rois 130 --num_spatial_boxes 1 --start 372

