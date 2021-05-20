#!/bin/bash
 
 EXE="./runpy.sh"
 NUM_GPUS=4
 LOG_DIR="./logs"
 
 echo "Enter a run name: (no spaces or slashes)"
 read RUN_NAME

# name run 
 EXPERIMENT_HASH=$(date +%s%N | sha256sum | head -c 6)
 EXP_NAME=${RUN_NAME}_${EXPERIMENT_HASH}
 OUTPUT_DIR="${LOG_DIR}/${EXP_NAME}"
 mkdir -p $OUTPUT_DIR
 
 echo "Running new experiments: $EXP_NAME"

 arg_queue=()
 for batch in 50 100 200; do
     for epochs in 20 25 40; do
        for lr in 0.001 .0005; do
            for layers in 6 12 18 ; do
                arg_queue+=("${batch} ${epochs} ${lr} ${layers}")
            done
        done
     done
 done

# run queue 
 job_num=0
 device_num=0
 for args in "${arg_queue[@]}"; do
 
     export CUDA_VISIBLE_DEVICES="$device_num"
 
     job_start=$(date +"%m-%d %H:%M:%S")
     echo $job_num $job_start $EXE $args | sed "s/ /, /g" >> ${OUTPUT_DIR}/job_list.csv
     echo "Launched $job_num @ $job_start on $device_num"
 
     $($EXE $args > ${OUTPUT_DIR}/${job_num}.log) &
 
     ((device_num++))
     ((job_num++))

     # yeah ok
     if [[ ${device_num} -eq ${NUM_GPUS} ]]; then
         wait
         device_num=0
     fi
 done

