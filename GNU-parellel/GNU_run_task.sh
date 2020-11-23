#!/bin/bash
export QT_API=pyside2
python -m cidan --headless True -ll error -p $1/parameters.json
rm -r $1/temp_files
rm -r $1/eigen_vectors