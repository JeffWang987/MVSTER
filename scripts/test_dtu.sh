#!/usr/bin/env bash
DTU_TESTPATH="/mnt/cfs/algorithm/public_data/mvs/dtu_test"
DTU_TESTLIST="lists/dtu/test.txt"

DTU_size=$1
exp=$2
PY_ARGS=${@:3}

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
DTU_CKPT_FILE=$DTU_LOG_DIR"/finalmodel.ckpt"
DTU_OUT_DIR="./outputs/dtu/"$exp



if [ $DTU_size = "raw" ] ; then
python test_mvs4.py --dataset=general_eval4 --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $DTU_OUT_DIR\
             --use_raw_train --thres_view 4 --conf 0.5 --group_cor --attn_temp 2 --inverse_depth $PY_ARGS | tee -a $DTU_LOG_DIR/log_test.txt
else
python test_mvs4.py --dataset=general_eval4 --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $DTU_OUT_DIR\
             --thres_view 4 --conf 0.5 --group_cor --attn_temp 2 --inverse_depth $PY_ARGS | tee -a $DTU_LOG_DIR/log_test.txt
fi
