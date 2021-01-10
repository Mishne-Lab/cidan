#/bin/bash
python eigen_display.py --e_dir neurofinder2.0/1/ --out_file plots/neurofinder2_eigen1.png --shape "(490,490)" --percent 94
python eigen_display.py --e_dir neurofinder2.0/2/ --out_file plots/neurofinder2_eigen2.png --shape "(490,490)" --percent 94
python multiblob.py --e_dir kdan/eigen_vectors --shape "(128,128)" \
--data_1 "kdan/roi_list.json" --data_2 "kdan/KDA79_A_keep121.json" --out_file plots/kdan_blob.png
python multiblob.py --e_dir kdan/eigen_vectors --shape "(128,128)" --blobs False \
--data_1 "kdan/roi_list.json" --data_2 "kdan/KDA79_A_keep121.json" --out_file plots/kdan_edge.png
python multiblob.py --e_dir kdan/eigen_vectors --shape "(128,128)" --blobs False \
--data_1 "kdan/roi_list.json"  --out_file plots/kdan_edge_cidan.png --color_1 "(255,255,255)"
python multiblob.py --e_dir kdan/eigen_vectors --shape "(128,128)" --blobs False \
--data_2 "kdan/KDA79_A_keep121.json"  --out_file plots/kdan_edge_caimain.png --color_2 "(255,255,255)"
python multiblob.py --e_dir demo_files/eigen_vectors  --blobs False \
--data_2 "demo_files/roi_list2.json" --data_1 "demo_files/roi_list1.json" \
--bg_path demo_files/embedding_norm_image.png --out_file plots/demo_edge.png
python multiblob.py --e_dir demo_files/eigen_vectors  --blobs True \
--data_2 "demo_files/roi_list2.json" --data_1 "demo_files/roi_list1.json" \
--bg_path demo_files/embedding_norm_image.png --out_file plots/demo_blob.png
python multiblob.py --e_dir demo_files/eigen_vectors  --blobs False \
--data_1 "demo_files/roi_list1.json" --color_1 "(255,255,255)" \
--bg_path demo_files/embedding_norm_image.png --out_file plots/demo_edge_1.png
python multiblob.py --e_dir demo_files/eigen_vectors  --blobs False \
--data_1 "demo_files/roi_list2.json" --color_1 "(255,255,255)" \
--bg_path demo_files/embedding_norm_image.png --out_file plots/demo_edge_2.png