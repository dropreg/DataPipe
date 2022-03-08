

## Multi-Teacher Distillation with Single Model for Neural Machine Translation

In this paper, we propose a simple yet effective knowledge distillation method to mimic multiple teacher distillation from the sub-network space and permuted variants of one single teacher model. we train a teacher by multiple sub-network extraction paradigms: sub-layer Reordering,  Layer-drop, and Dropout variants (RLD). In doing so, one teacher model can provide multiple outputs variants and causes neither additional parameters nor much extra training cost.

## How to use

1. First download and preprocess the data (IWSLT'14 German to English https://github.com/pytorch/fairseq/tree/main/examples/translation)
2. Next we'll train a RLD teacher model over this data:
   ```
       # set flag to train
       TRAIN_FLAG=true
       TEST_FLAG=false
       bash script/train_tec/run_sld_tec.sh
   ```
3. Generate Distilled data by trained teacher model:
   > The generated data needs to be reprocessed to binary files like step 1.
   ```
       # set flag to inference
       TRAIN_FLAG=false
       TEST_FLAG=true
       bash script/train_tec/run_sld_tec.sh
   ```
4. Finally we can train our student model over generated data:
   ```
       bash script/train_stu/run_datakd_stu.sh
   ```

## Ablation study

You can reproduce our ablation experiments by change the hyper-paramter
  ```
      # layer-reorder
      --sublayer-reorder
      # layer-dropout
      --sublayer-drop 
      --encoder-sub-layerdrop 0.2 --decoder-sub-layerdrop 0.2 
      # model dropout
      --dropout 0.3 
  ```



