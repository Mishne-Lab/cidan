#!/bin/bash
export QT_API=pyside2
python -m cidan --headless True -ll debug -p $1/parameters.json -lp $1/log.txt
rm -r $1/temp_files/
#rm -r $1/eigen_vectors
