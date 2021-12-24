This repo for paper: Multi-Teacher Distillation with Single Model for Neural Machine Translation

# how to generate different submodules:

## Dropout

1. dropout

train teacher model
> bash script/run_baseline.sh

change script train&test flag to generate teacher data

> bash script/train_tec/run_baseline.sh
> bash script/generate_data/process_gen.sh /data/lxb/sm_gen_data/de-en-data/baseline

2. Layer Drop

train teacher model
> bash script/run_ld_tec.sh

3. sub-Layer Reorder & Drop

train teacher model
> bash script/run_sld_tec.sh

param:
--train-tec \
--sublayer-reorder \
--sublayer-drop \
--encoder-sub-layerdrop 0.2 --decoder-sub-layerdrop 0.2 

## distillation

1. label based kd

train student model with mutli dataset
> bash script/train_stu/run_labeledkd_stu.sh

2. data based kd

train student model with mutli dataset
> bash script/train_stu/run_multi_stu.sh


train student model with mutli dataset
> bash script/train_stu/run_datakd_stu.sh
