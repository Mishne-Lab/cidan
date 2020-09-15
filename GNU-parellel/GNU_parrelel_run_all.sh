parallel --jobs 1  --joblog task_file2_special.log  --resume --progress \
               --colsep ',' -a task_list.csv ./GNU_run_task.sh {}
