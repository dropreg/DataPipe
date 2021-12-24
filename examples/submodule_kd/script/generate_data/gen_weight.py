import os
import sacrebleu
from tqdm import tqdm
import pickle as pkl
import numpy as np


def check_duplocation(file_num):
    # data_dir = "/data/lxb/gen_data/kd_data/checkpoint_tec"
    # data_dir = "/data/lxb/gen_data/kd_data/rld_drop"
    # data_dir = "/data/lxb/gen_data/kd_data/rld_tec"
    # data_dir = "/data/lxb/gen_data/kd_data/drop_var_tec"
    # data_dir = "/data/lxb/gen_data/kd/rld_tec_new/order_base"
    data_dir = "/data/lxb/sm_gen_data/de-en-data/baseline_new/"
    
    gen_files = [os.path.join(data_dir, "de-en.en_{}".format(index)) for index in range(file_num)]
    sent_set = set()
    for gen_file in gen_files:
        for gen_line in open(gen_file):
            gen_sent = gen_line.strip().replace("@@ ", "")
            if gen_sent not in sent_set:
                sent_set.add(gen_sent)
    print(len(sent_set))

def gen_belu_weight(file_num):
    
    data_dir = "/data/lxb/gen_data/kd_data/drop_var_tec"
    ground_truth_file = os.path.join(data_dir, "en_8.txt") 
    gen_files = [os.path.join(data_dir, "en_{}.txt".format(index)) for index in range(file_num)]
    
    best_score = {}
    for gen_file in gen_files:
        line_num = 0
        for (gd_line, gen_line) in tqdm(zip(open(ground_truth_file).readlines(), open(gen_file).readlines())):
            gd_sent = gd_line.strip().replace("@@ ", "")
            gen_sent = gen_line.strip().replace("@@ ", "")
            sent_bleu_score = sacrebleu.sentence_bleu([gd_sent], [gen_sent]).score / 100.
            if line_num not in best_score:
                best_score[line_num] = [sent_bleu_score]
            else:
                best_score[line_num].append(sent_bleu_score)
            line_num += 1
    pkl.dump(best_score, open("bleu_score.pkl", 'wb'))


def gen_best_sentence(file_num):
    
    data_dir = "/data/lxb/gen_data/kd_data/rld_tec"
    ground_truth_file = os.path.join(data_dir, "en_6.txt") 
    gen_files = [os.path.join(data_dir, "en_{}.txt".format(index)) for index in range(file_num)]
    
    best_score = {}
    best_sentence = {}
    for gen_file in gen_files:
        line_num = 0
        for (gd_line, gen_line) in tqdm(zip(open(ground_truth_file).readlines(), open(gen_file).readlines())):
            gd_sent = gd_line.strip().replace("@@ ", "")
            gen_sent = gen_line.strip().replace("@@ ", "")
            sent_bleu_score = sacrebleu.sentence_bleu([gd_sent], [gen_sent]).score / 100.
            
            if line_num in best_score:
                if sent_bleu_score > best_score[line_num]:
                    best_score[line_num] = sent_bleu_score
                    best_sentence[line_num] = gen_line
            else:
                best_score[line_num] = sent_bleu_score
                best_sentence[line_num] = gen_line
            line_num += 1
    
    with open("/data/lxb/gen_data/kd_data/rld_best/en_0.txt", 'w') as fo:
        for line_num, sent in best_sentence.items():
            fo.writelines(sent)

def gen_single_sentence(file_num):
    # data_dir = "/data/lxb/gen_data/kd_data/rld_tec"
    data_dir = "/data/lxb/gen_data/kd_data/rld_tec"
    de_file = os.path.join(data_dir, "de.txt")
    gen_files = [os.path.join(data_dir, "en_{}.txt".format(index)) for index in range(file_num)]
    
    pair_score = set()
    for gen_file in gen_files:
        line_num = 0
        for (de_line, gen_line) in tqdm(zip(open(de_file).readlines(), open(gen_file).readlines())):
            if (de_line, gen_line) not in pair_score:
                pair_score.add((de_line, gen_line))
            line_num += 1
    print(len(pair_score))
    with open("/data/lxb/gen_data/kd_data/single_tec/en_0.txt", 'w') as fo_t, open("/data/lxb/gen_data/kd_data/single_tec/de_0.txt", 'w') as fo_s:
        for (de_line, gen_line) in pair_score:
            fo_s.writelines(de_line)
            fo_t.writelines(gen_line)

total_num = 5
# gen_single_sentence(total_num)
# gen_best_sentence(total_num)
check_duplocation(total_num)
# gen_belu_weight(total_num)


def check_id(file_num):
    rld_data_dir = "/data/lxb/gen_data/kd/rld_tec_new/order_base"
    drop_data_dir = "/data/lxb/gen_data/kd/rld_tec_new/"

    gen_files = [os.path.join(rld_data_dir, "en_{}.txt".format(index)) for index in range(file_num)]
    rld_sent_set = {}
    
    for gen_file in gen_files:
        line_num = 0
        for gen_line in open(gen_file):
            gen_sent = gen_line.strip().replace("@@ ", "")
            
            if line_num in rld_sent_set:
                rld_sent_set[line_num].add(gen_sent)
            else:
                rld_sent_set[line_num] = set()
                rld_sent_set[line_num].add(gen_sent)
            line_num += 1
    
    gen_files = [os.path.join(drop_data_dir, "en_{}.txt".format(index)) for index in range(file_num)]
    drop_sent_set = {}
    for gen_file in gen_files:
        line_num = 0
        for gen_line in open(gen_file):
            gen_sent = gen_line.strip().replace("@@ ", "")
            
            if line_num in drop_sent_set:
                drop_sent_set[line_num].add(gen_sent)
            else:
                drop_sent_set[line_num] = set()
                drop_sent_set[line_num].add(gen_sent)
            line_num += 1
    
    num = 0
    rld_num = 0
    drop_num = 0
    fix_num = 1
    for i in range(line_num):
        if len(rld_sent_set[i]) > fix_num:
            rld_num += 1
        if len(drop_sent_set[i]) > fix_num:
            drop_num += 1
        if len(rld_sent_set[i]) > fix_num and len(drop_sent_set[i]) > fix_num:
            num += 1
    print(num, rld_num, drop_num)
    print(len(rld_sent_set), len(drop_sent_set))

# check_id(6)
