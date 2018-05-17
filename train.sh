#!/bin/bash

TRAIN_TEST_FLAG=1

if [ ${TRAIN_TEST_FLAG} -eq 1  ]; then
    echo "\n---------TRAIN STAGE---------\n"
    ROOT_FOLDER="/home/hzwangjialei/project/pytorch_train/ICDARTrain"
    BG_FOLDER="./BG_FOLDER"
    SAVE_PATH="torch_v2_pc_epoch40"
    RESUME_FLAG=0
    # RESUME_PATH="./pretrain_models/resnet34_real_oriloss_249.ckpt"
    RESUME_PATH="./torch_v2_sy8/torch_v2_sy8_50.ckpt"
    LERNING_RATE=0.00004
    LOSS_RATIO=2.0
    #BLOCKS="2,3,5,3"
    GPUS="0"
    PHASE="train"
    START_EPOCH=0
    END_EPOCH=40
    BATCH_SIZE=32
    NUM_WORKERS=16

    LOG_ID=$(date -d "today" +"%Y%m%d_%H%M%S")
    SAVE_LOG_PATH="./${SAVE_PATH}/${LOG_ID}"
    SAVE_LOG_NAME="./${SAVE_LOG_PATH}/log.${LOG_ID}.txt"
    if [ -d ${SAVE_PATH} ]; then
        echo "${SAVE_PATH} EXSITS"
    else
        mkdir ${SAVE_PATH}
    fi
    if [ -d ${SAVE_LOG_PATH} ]; then
        echo "${SAVE_LOG_PATH} EXSITS"
    else
        mkdir ${SAVE_LOG_PATH}
    fi

    cp train.sh  ${SAVE_LOG_PATH}/
    cp run.py  ${SAVE_LOG_PATH}/
    #cp net_origin.py ${SAVE_LOG_PATH}/
    #cp preprocess.py ${SAVE_LOG_PATH}/

    python -u run.py  \
        --root_folder=${ROOT_FOLDER} \
        --bg_folder=${BG_FOLDER} \
        --save_folder=${SAVE_PATH} \
        --gpu_list=${GPUS} \
        --phase=${PHASE} \
        --start_epoch=${START_EPOCH} \
        --end_epoch=${END_EPOCH} \
        --batch_size=${BATCH_SIZE} \
        --num_workers=${NUM_WORKERS} \
        --learning_rate=${LERNING_RATE} \
        --blocks=${BLOCKS} \
        --loss_ratio=${LOSS_RATIO} \
        --resume_flag=${RESUME_FLAG} \
        --resume_path=${RESUME_PATH} \
        2>&1| tee  ${SAVE_LOG_NAME}
    echo "THE PROGRAM ENTERING BACKGROUND"
    echo "See Logs By Using Command like below"
    echo "tail -f ${SAVE_LOG_NAME}"
fi
