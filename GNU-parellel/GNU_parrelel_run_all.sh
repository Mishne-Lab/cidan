parallel --jobs 1 --slf hostfile  --joblog task2.log  --resume --timeout 2500 --progress \
               --colsep ',' -a task_list.csv ./GNU_run_task.sh {}