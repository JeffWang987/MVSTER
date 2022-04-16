#!/usr/bin/env bash
DTU_TRAINING="/mnt/cfs/algorithm/public_data/mvs/mvs_training/dtu"
DTU_TRAINLIST="lists/dtu/train.txt"
DTU_TESTLIST="lists/dtu/test.txt"

DTU_trainsize=$1
exp=$2
PY_ARGS=${@:3}

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi

DTU_CKPT_FILE=$DTU_LOG_DIR"/finalmodel.ckpt"
DTU_OUT_DIR="./outputs/dtu/"$exp


if [ $DTU_trainsize = "raw" ] ; then
python -m torch.distributed.launch --nproc_per_node=4 train_mvs4.py --logdir $DTU_LOG_DIR --dataset=dtu_yao4 --batch_size=2 --trainpath=$DTU_TRAINING --summary_freq 100 \
                --group_cor --inverse_depth --rt --mono --attn_temp 2 --use_raw_train --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST  $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt
else
python -m torch.distributed.launch --nproc_per_node=4 train_mvs4.py --logdir $DTU_LOG_DIR --dataset=dtu_yao4 --batch_size=2 --trainpath=$DTU_TRAINING --summary_freq 100 \
                --group_cor --inverse_depth --rt --mono --attn_temp 2 --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST  $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt
fi

