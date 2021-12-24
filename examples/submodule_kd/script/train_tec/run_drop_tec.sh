#!/bin/bash

TRAIN_FLAG=false
TEST_FLAG=false

SRC_LNG=de
TGT_LNG=en

GPU=0

export CUDA_VISIBLE_DEVICES=$GPU

ROOT_DIR=/data/lxb
DATA_DIR=$ROOT_DIR/dataset/small_nmt_data/iwslt_data/${SRC_LNG}-${TGT_LNG}_file/databin
CKPT_DIR=$ROOT_DIR/checkpoints/submodule_kd/${SRC_LNG}-${TGT_LNG}-ckpt/drop_all_tec

mkdir -p $CKPT_DIR

if [ $TRAIN_FLAG == true ]; then
    
    echo "[SubModule_KD info]: train nmt model from lang $SRC_LNG to $TGT_LNG "
    rm $CKPT_DIR/train.log
    touch $CKPT_DIR/train.log
    
    fairseq-train $DATA_DIR \
        --user-dir examples/submodule_kd/kd_src \
        --arch x_transformer \
        --task kd_translation \
        --train-tec \
        -s $SRC_LNG -t $TGT_LNG \
        --share-all-embeddings \
        --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
        --optimizer adam --lr 0.0005 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
        --label-smoothing 0.1 --criterion kd_label_smoothed_cross_entropy \
        --max-update 100000 --warmup-updates 4000 --max-tokens 4096 \
        --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
        --no-progress-bar \
        --no-epoch-checkpoints \
        --keep-best-checkpoints 10 \
        --fp16 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --save-dir $CKPT_DIR | tee -a $CKPT_DIR/train.log \

elif [ $TEST_FLAG == true ]; then

    echo "[SubModule_KD info]: test nmt model from lang $SRC_LNG to $TGT_LNG "

    if [ $SRC_LNG == de ] || [ $TGT_LNG == de ]; then
        
        # fairseq evaluation code
        fairseq-generate $DATA_DIR \
            --user-dir examples/submodule_kd/kd_src \
            --task kd_translation \
            -s $SRC_LNG -t $TGT_LNG \
            --path $CKPT_DIR/checkpoint_best.pt \
            --batch-size 128 --beam 5 --remove-bpe --quiet \
        
        # fairseq-generate $DATA_DIR \
        #     -s $SRC_LNG -t $TGT_LNG \
        #     --path $CKPT_DIR/checkpoint_best.pt \
        #     --encoder-layers-to-keep 0,2,4 --decoder-layers-to-keep 0,2,4
        #     --batch-size 128 --beam 5 --remove-bpe > gen.out

        # bash ../../scripts/compound_split_bleu.sh gen.out
        # rm gen.out

    else

        BPE_IN=$DATA_DIR/../test.$SRC_LNG
        cat $BPE_IN | fairseq-interactive $DATA_DIR \
            --source-lang $SRC_LNG --target-lang $TGT_LNG \
            --path $CKPT_DIR/checkpoint_best.pt \
            --buffer-size 2000 --batch-size 128 \
            --beam 5 --remove-bpe > iwslt17.test.$SRC_LNG-$TGT_LNG.$TGT_LNG.sys
        grep ^H iwslt17.test.$SRC_LNG-$TGT_LNG.$TGT_LNG.sys | cut -f3 > out.log
        sacremoses -l $TGT_LNG detokenize < out.log > deout.log
        cat deout.log | sacrebleu $BPE_REF --language-pair $SRC_LNG-$TGT_LNG
        
        rm iwslt17.test.$SRC_LNG-$TGT_LNG.$TGT_LNG.sys
        rm out.log
        rm deout.log

    fi

else

    echo "[SubModule_KD info]: generate KD data from lang $SRC_LNG to $TGT_LNG "
    GEN_DIR=$ROOT_DIR/sm_gen_data/${SRC_LNG}-${TGT_LNG}-data/drop_all_tec_daopen_new
    mkdir -p $GEN_DIR
    
    SEED=8

    fairseq-generate $DATA_DIR \
        --user-dir examples/submodule_kd/kd_src \
        --task kd_translation \
        -s $SRC_LNG -t $TGT_LNG \
        --gen-subset 'train' \
        --generate-random $SEED \
        --path $CKPT_DIR/checkpoint_best.pt \
        --batch-size 128 --beam 5 > $GEN_DIR/${SRC_LNG}_${TGT_LNG}_$SEED.log

fi
