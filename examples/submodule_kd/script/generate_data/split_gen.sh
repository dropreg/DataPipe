#!/bin/bash


SRC_LNG=en
TGT_LNG=de
GEN_DIR=/data/lxb/sm_gen_data/${SRC_LNG}-${TGT_LNG}-data/baseline_new


for INDEX in {2..8}
do

    cp -r $GEN_DIR/databin  $GEN_DIR/databin_$INDEX

    cd $GEN_DIR/databin_$INDEX

    for REMOVE_INDEX in $(seq ${INDEX} 8)
    do
        
        rm *$REMOVE_INDEX*
    
    done

done