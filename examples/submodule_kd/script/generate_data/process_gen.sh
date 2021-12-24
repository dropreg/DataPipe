#!/bin/bash

SRC_LNG=de
TGT_LNG=en

LANG=${SRC_LNG}-${TGT_LNG}
echo "[SubModule_KD info]: prepare data from lang $SRC_LNG to $TGT_LNG "

ROOT_DIR=/data/lxb
if [ $SRC_LNG == en ]; then
    DATA_DIR=$ROOT_DIR/dataset/small_nmt_data/iwslt_data/${TGT_LNG}-${SRC_LNG}_file/databin
else
    DATA_DIR=$ROOT_DIR/dataset/small_nmt_data/iwslt_data/${SRC_LNG}-${TGT_LNG}_file/databin
fi

# check the log file dir
if [ ${1} ]; then
    
    echo "[SubModule_KD info]: accept the dir parameter ${1}"
    GEN_DIR=${1}

else
        
    # GEN_DIR=$ROOT_DIR/sm_gen_data/${SRC_LNG}-${TGT_LNG}-data/baseline
    # baseline_td_new
    # GEN_DIR=/data/lxb/sm_gen_data/${SRC_LNG}-${TGT_LNG}-data/baseline_ad_new
    GEN_DIR=/data/lxb/sm_gen_data/de-en-data/test_mask/

fi

if [ ! -d $GEN_DIR ]; then

    echo "[SubModule_KD info]: $GEN_DIR does't exist "
    exit

fi

GEN_BIN_DIR=$GEN_DIR/databin
mkdir -p $GEN_BIN_DIR

echo "[SubModule_KD info]: copy dict from dir: ${DATA_DIR} "
cp ${DATA_DIR}/dict.${SRC_LNG}.txt $GEN_BIN_DIR
cp ${DATA_DIR}/dict.${TGT_LNG}.txt $GEN_BIN_DIR

# you need to add generated files
LOG_FILES=($GEN_DIR/${SRC_LNG}_${TGT_LNG}_*.log)
FILE_IDX=0
for LOG_FILE in ${LOG_FILES[*]}; do

    echo "[SubModule_KD info]: extract data from log file $LOG_FILE with index ${FILE_IDX}"

    if [ ! -f ${LANG}.${SRC_LNG} ]; then

        cat $LOG_FILE | grep ^S | cut -f2 > $GEN_DIR/${LANG}.${SRC_LNG}
        cat $LOG_FILE | grep ^T | cut -f2 > $GEN_DIR/${LANG}.${TGT_LNG}

    fi

    cat $LOG_FILE | grep ^H | cut -f3 > $GEN_DIR/${LANG}.${TGT_LNG}_${FILE_IDX}
    BPE_FILES+=($GEN_DIR/${LANG}.${TGT_LNG}_${FILE_IDX})
    
    if [ ! -f $GEN_DIR/${LANG}.${TGT_LNG}_${FILE_IDX} ]; then

        echo "[SubModule_KD info]: ${LANG}.${TGT_LNG}_${FILE_IDX} create error! "
        exit

    fi
    let FILE_IDX=$FILE_IDX+1

done

FILE_IDX=0
for BPE_FILE in ${BPE_FILES[*]}; do

    echo "[SubModule_KD info]: build binary data from bpe file $BPE_FILE with index ${FILE_IDX}"

    fairseq-preprocess --source-lang $SRC_LNG --target-lang ${TGT_LNG}_${FILE_IDX} \
        --trainpref $GEN_DIR/${LANG} \
        --destdir $GEN_BIN_DIR \
        --srcdict $GEN_BIN_DIR/dict.${SRC_LNG}.txt \
        --tgtdict $GEN_BIN_DIR/dict.${TGT_LNG}.txt \
        --workers 20 \
    
    rm $GEN_BIN_DIR/dict.${TGT_LNG}_${FILE_IDX}.txt
    
    let FILE_IDX_INCRMENT=${FILE_IDX}+1
    mv $GEN_BIN_DIR/train.${LANG}_${FILE_IDX}.${TGT_LNG}_${FILE_IDX}.idx $GEN_BIN_DIR/train${FILE_IDX_INCRMENT}.${LANG}.${TGT_LNG}.idx
    mv $GEN_BIN_DIR/train.${LANG}_${FILE_IDX}.${SRC_LNG}.idx $GEN_BIN_DIR/train${FILE_IDX_INCRMENT}.${LANG}.${SRC_LNG}.idx
    mv $GEN_BIN_DIR/train.${LANG}_${FILE_IDX}.${TGT_LNG}_${FILE_IDX}.bin $GEN_BIN_DIR/train${FILE_IDX_INCRMENT}.${LANG}.${TGT_LNG}.bin 
    mv $GEN_BIN_DIR/train.${LANG}_${FILE_IDX}.${SRC_LNG}.bin $GEN_BIN_DIR/train${FILE_IDX_INCRMENT}.${LANG}.${SRC_LNG}.bin 
    let FILE_IDX=${FILE_IDX}+1

done

echo "[SubModule_KD info]: copy binary data from ${DATA_DIR} to ${GEN_BIN_DIR}"
cp ${DATA_DIR}/*.idx ${GEN_BIN_DIR}
cp ${DATA_DIR}/*.bin ${GEN_BIN_DIR}