#!/bin/bash
#python eigen_display.py --e_dir data/neurofinder2.0/1/ --out_file plots/neurofinder2_eigen1.png --shape "(490,490)" --percent 94
#python eigen_display.py --e_dir data/neurofinder2.0/2/ --out_file plots/neurofinder2_eigen2.png --shape "(490,490)" --percent 94
#python multiblob.py --e_dir data/kdan/eigen_vectors --shape "(128,128)" \
#--data_1 "data/kdan/roi_list.json" --data_2 "data/kdan/KDA79_A_keep121.json" --out_file plots/kdan_blob.png
#python multiblob.py --e_dir data/kdan/eigen_vectors --shape "(128,128)" --blobs False \
#--data_1 "data/kdan/roi_list.json" --data_2 "data/kdan/KDA79_A_keep121.json" --out_file plots/kdan_edge.png
#python multiblob.py --e_dir data/kdan/eigen_vectors --shape "(128,128)" --blobs False \
#--data_1 "data/kdan/roi_list.json"  --out_file plots/kdan_edge_cidan.png --color_1 "(255,255,255)"
#python multiblob.py --e_dir data/kdan/eigen_vectors --shape "(128,128)" --blobs False \
#--data_2 "data/kdan/KDA79_A_keep121.json"  --out_file plots/kdan_edge_caimain.png --color_2 "(255,255,255)"
#python multiblob.py --e_dir data/demo_files/eigen_vectors  --blobs False \
#--data_2 "data/demo_files/roi_list2.json" --data_1 "data/demo_files/roi_list1.json" \
#--bg_path data/demo_files/embedding_norm_image.png --out_file plots/demo_edge.png
#python multiblob.py --e_dir data/demo_files/eigen_vectors  --blobs True \
#--data_2 "data/demo_files/roi_list2.json" --data_1 "data/demo_files/roi_list1.json" \
#--bg_path data/demo_files/embedding_norm_image.png --out_file plots/demo_blob.png
#python multiblob.py --e_dir data/demo_files/eigen_vectors  --blobs False \
#--data_1 "data/demo_files/roi_list1.json" --color_1 "(255,255,255)" \
#--bg_path data/demo_files/embedding_norm_image.png --out_file plots/demo_edge_1.png
#python multiblob.py --e_dir data/demo_files/eigen_vectors  --blobs False \
#--data_1 "data/demo_files/roi_list2.json" --color_1 "(255,255,255)" \
#--bg_path data/demo_files/embedding_norm_image.png --out_file plots/demo_edge_2.png
python run_multiblob.py --data_1 /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan \
--data_2 /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/caiman \
--data_3 /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/suite2p \
--data_4 /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/true
#
#mkdir plots/timeFile1
#python time_graph.py --time_trace_path /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File1_CPn_l5_gcamp6s_lan.tif300/time_traces.pickle --out plots/timeFile1/File1.png
#mkdir plots/timeFile2
#python time_graph.py --time_trace_path /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File2_CSt_l5_gcamp6s_lan.tif238/time_traces.pickle --out plots/timeFile2/File2.png
#mkdir plots/timeFile3
#python time_graph.py --time_trace_path /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File3_kf139_s.tif115/time_traces.pickle --out plots/timeFile3/File3.png
#mkdir plots/timeFile4
#python time_graph.py --time_trace_path /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File4_kf158_s.tif429/time_traces.pickle --out plots/timeFile4/File4.png
#mkdir plots/timeFile5
#python time_graph.py --time_trace_path /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File5_l23_gcamp6s_lan.tif26/time_traces.pickle --out plots/timeFile5/File5.png
#mkdir plots/timeFile6
#python time_graph.py --time_trace_path /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File6_som_l5_gcamp6s_alka.tif836/time_traces.pickle --out plots/timeFile6/File6.png
cp -r plots/* visualizer/web/images
python visualizer/html.py



