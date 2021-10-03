#/bin/bash
rm task_list.csv
python createTaskList.py -idir ~/Desktop/HigleyData/File1_p/ -odir ~/Desktop/HigleyDataOutPos2 -p ~/Desktop/CIDAN/poisson/parameters/File1.json
python createTaskList.py -idir ~/Desktop/HigleyData/File2_p/ -odir ~/Desktop/HigleyDataOutPos2 -p ~/Desktop/CIDAN/poisson/parameters/File2.json
python createTaskList.py -idir ~/Desktop/HigleyData/File3_p/ -odir ~/Desktop/HigleyDataOutPos2 -p ~/Desktop/CIDAN/poisson/parameters/File3.json
python createTaskList.py -idir ~/Desktop/HigleyData/File4_p/ -odir ~/Desktop/HigleyDataOutPos2 -p ~/Desktop/CIDAN/poisson/parameters/File4.json
python createTaskList.py -idir ~/Desktop/HigleyData/File5_p/ -odir ~/Desktop/HigleyDataOutPos2 -p ~/Desktop/CIDAN/poisson/parameters/File5.json
python createTaskList.py -idir ~/Desktop/HigleyData/File6_p/ -odir ~/Desktop/HigleyDataOutPos2 -p ~/Desktop/CIDAN/poisson/parameters/File6.json
