#!/bin/bash

TRAIN_FLAG=true
KEEP_DIR=false

SRC_LNG=de
TGT_LNG=en

export CUDA_VISIBLE_DEVICES=0

ROOT_DIR=/data/lxb
DATA_DIR=$ROOT_DIR/dataset/small_nmt_data/iwslt_data/${SRC_LNG}-${TGT_LNG}_file/databin
CKPT_DIR=$ROOT_DIR/checkpoints/submodule_kd/${SRC_LNG}-${TGT_LNG}-ckpt/tec
TEC_WEIGHT=$ROOT_DIR/checkpoints/submodule_kd/${SRC_LNG}-${TGT_LNG}-ckpt/tec_weight.pkl
mkdir -p $CKPT_DIR

if [ $TRAIN_FLAG == true ]; then

    echo "[SubModule_KD info]: train nmt model from lang $SRC_LNG to $TGT_LNG "
    rm $CKPT_DIR/train.log
    touch $CKPT_DIR/train.log
    
    python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 \
        ../../../fairseq_cli/train.py $DATA_DIR \
        --ddp-backend=no_c10d \
        --load-tec-weight $TEC_WEIGHT \
        --user-dir examples/submodule_kd/kd_src -s $SRC_LNG -t $TGT_LNG \
        --task kd_translation \
        --share-all-embeddings \
        --upsample-primary 1 \
        --dropout 0.3 \
        --optimizer adam --lr 0.0005 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
        --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
        --max-update 100000 --warmup-updates 4000 --max-tokens 4096 \
        --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
        --no-progress-bar \
        --keep-best-checkpoints 10 \
        --no-epoch-checkpoints \
        --fp16 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --save-dir $CKPT_DIR | tee -a $CKPT_DIR/train.log \
    
else

    echo "[SubModule_KD info]: test nmt model from lang $SRC_LNG to $TGT_LNG "

    if [ $SRC_LNG == de ] || [ $TGT_LNG == de ]; then
        
        # fairseq evaluation code
        # fairseq-generate $DATA_DIR \
        #     -s $SRC_LNG -t $TGT_LNG \
        #     --path $CKPT_DIR/checkpoint_best.pt \
        #     --batch-size 128 --beam 5 --remove-bpe --quiet \

        fairseq-generate $DATA_DIR \
            -s $SRC_LNG -t $TGT_LNG \
            --path $CKPT_DIR/checkpoint_best.pt \
            --batch-size 128 --beam 5 --remove-bpe > gen.out

        bash ../../scripts/compound_split_bleu.sh gen.out
        rm gen.out

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

fi
