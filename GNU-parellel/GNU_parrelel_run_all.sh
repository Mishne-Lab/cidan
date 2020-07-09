parallel --jobs 1 --slf hostfile  --joblog task2.log  --resume --progress \
               --colsep ',' -a task_list.csv ./GNU_run_task.sh {}