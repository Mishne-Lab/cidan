#/bin/bash
mkdir ~/Desktop/HigleyData/File1_p/
mkdir ~/Desktop/HigleyData/File2_p/
mkdir ~/Desktop/HigleyData/File3_p/
mkdir ~/Desktop/HigleyData/File4_p/
mkdir ~/Desktop/HigleyData/File5_p/
mkdir ~/Desktop/HigleyData/File6_p/
python apply_poisson.py --input_file ~/Desktop/HigleyData/file1/File1_CPn_l5_gcamp6s_lan.tif --output_file ~/Desktop/HigleyData/File1_p/File1.tif --number 4
python apply_poisson.py --input_file ~/Desktop/HigleyData/file2/File2_CSt_l5_gcamp6s_lan.tif --output_file ~/Desktop/HigleyData/File2_p/File2.tif --number 4
python apply_poisson.py --input_file ~/Desktop/HigleyData/file3/File3_kf139_s.tif --output_file ~/Desktop/HigleyData/File3_p/File3.tif --number 4
python apply_poisson.py --input_file ~/Desktop/HigleyData/file4/File4_kf158_s.tif --output_file ~/Desktop/HigleyData/File4_p/File4.tif --number 4
python apply_poisson.py --input_file ~/Desktop/HigleyData/file5/File5_l23_gcamp6s_lan.tif --output_file ~/Desktop/HigleyData/File5_p/File5.tif --number 4
python apply_poisson.py --input_file ~/Desktop/HigleyData/file6/File6_som_l5_gcamp6s_alka.tif --output_file ~/Desktop/HigleyData/File6_p/File6.tif --number 4
