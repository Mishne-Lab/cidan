#!/bin/bash
conda activate lab02
python -m CIDAN --headless True -ll error -p $1
rm -r $1/temp_files
rm -r $1/eigen_vectors