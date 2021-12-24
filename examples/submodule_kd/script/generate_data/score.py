import os
import sacrebleu
from tqdm import tqdm
import pickle as pkl
import numpy as np

baseline_data_dir = "/data/lxb/gen_data/base_data/baseline"

rld_data_dir = "/data/lxb/gen_data/kd_data/rld_tec"

mtdd_data_dir = "/data/lxb/sm_gen_data/de-en-data/baseline_new"

def gen_belu_weight(file_num):

    data_dir = "/data/lxb/gen_data/kd_data/kd_rld/de-en/rld_all_tec/"
    ground_truth_file = os.path.join(data_dir, "de_en.en_target") 
    gen_files = [os.path.join(data_dir, "en_{}.txt".format(index)) for index in range(file_num)]
    
    bleu_score = {}
    sentence_map = {}
    for gen_file in gen_files:
        line_num = 0
        for (gd_line, gen_line) in tqdm(zip(open(ground_truth_file).readlines(), open(gen_file).readlines())):
            gd_sent = gd_line.strip().replace("@@ ", "")
            gen_sent = gen_line.strip().replace("@@ ", "")
            sent_bleu_score = sacrebleu.sentence_bleu([gd_sent], [gen_sent]).score / 100.
            if line_num not in bleu_score:
                bleu_score[line_num] = [sent_bleu_score]
                sentence_map[line_num] = [gen_sent]
            else:
                bleu_score[line_num].append(sent_bleu_score)
                sentence_map[line_num].append(gen_sent)
            line_num += 1
    pkl.dump(bleu_score, open("rld_bleu_score.pkl", 'wb'))
    pkl.dump(sentence_map, open("rld_sentence_map.pkl", 'wb'))

def cacluate_self_bleu():
    sentence_map  = pkl.load(open("baseline_sentence_map.pkl", 'rb'))
    self_bleu = {}
    import pdb; pdb.set_trace()
    for line_num, sentence_list in sentence_map.items():
        diff_list = []
        for a_idx, a in enumerate(sentence_list):
            for b_idx, b in enumerate(sentence_list):
                if a_idx != b_idx:
                    sent_bleu_score = sacrebleu.sentence_bleu([a], [b]).score / 100.
                    diff_list.append(1 - sent_bleu_score)
        sentence_self_bleu = sum(diff_list) / (len(sentence_list) * (len(sentence_list) - 1))
        self_bleu[line_num] = sentence_self_bleu
    pkl.dump(self_bleu, open("baseline_self_bleu.pkl", 'wb'))

def read_self_bleu():
    sentence_map  = pkl.load(open("baseline_sentence_map.pkl", 'rb'))
    self_bleu  = pkl.load(open("baseline_self_bleu.pkl", 'rb'))
    import pdb; pdb.set_trace()


def get_max_bleu():
    bleu_score = pkl.load(open("baseline_bleu_score.pkl", 'rb'))
    sentence_map = pkl.load(open("baseline_sentence_map.pkl", 'rb'))
    max_num = 0
    max_list = [0] * 6
    for line_num, bleu_list in bleu_score.items():
        max_num += max(bleu_list)
        for b_idx, b in enumerate(bleu_list):
            max_list[b_idx] += b
    print(max_num)
    print(max_list)

def gen_rld_belu_weight(file_num):

    data_dir = "/data/lxb/gen_data/kd_data/kd_rld/zh-en/rld_all_tec"
    ground_truth_file = os.path.join(data_dir, "en_7.txt") 
    gen_files = [os.path.join(data_dir, "en_{}.txt".format(index)) for index in range(file_num)]
    
    bleu_score = {}
    for gen_file in gen_files:
        line_num = 0
        for (gd_line, gen_line) in tqdm(zip(open(ground_truth_file).readlines(), open(gen_file).readlines())):
            gd_sent = gd_line.strip().replace("@@ ", "")
            gen_sent = gen_line.strip().replace("@@ ", "")
            sent_bleu_score = sacrebleu.sentence_bleu([gd_sent], [gen_sent]).score / 100.
            if line_num not in bleu_score:
                bleu_score[line_num] = [sent_bleu_score]
            else:
                bleu_score[line_num].append(sent_bleu_score)
            line_num += 1
    pkl.dump(bleu_score, open("pkl_file/zh_en_rld.pkl", 'wb'))




total_num = 7
# gen_rld_belu_weight(total_num)
# cacluate_self_bleu()
# read_self_bleu()
# get_max_bleu()

def gen_lxb_bleu():
    data_dir = "/data/lxb/sm_gen_data/de-en-data/baseline_new"
    ground_truth_file = os.path.join(data_dir, "de_en.en_target")
    gen_files = [os.path.join(data_dir, "de-en.en_{}".format(index)) for index in range(5)]
    
    bleu_score = {}
    sentence_map = {}
    for gen_file in gen_files:
        line_num = 0
        for (gd_line, gen_line) in tqdm(zip(open(ground_truth_file).readlines(), open(gen_file).readlines())):
            gd_sent = gd_line.strip().replace("@@ ", "")
            gen_sent = gen_line.strip().replace("@@ ", "")
            sent_bleu_score = sacrebleu.sentence_bleu([gd_sent], [gen_sent]).score / 100.
            if line_num not in bleu_score:
                bleu_score[line_num] = [sent_bleu_score]
                sentence_map[line_num] = [gen_sent]
            else:
                bleu_score[line_num].append(sent_bleu_score)
                sentence_map[line_num].append(gen_sent)
            line_num += 1
    pkl.dump(bleu_score, open("bleu_score.pkl", 'wb'))
    pkl.dump(sentence_map, open("sentence_map.pkl", 'wb'))

def lxb_self_bleu():
    sentence_map  = pkl.load(open("sentence_map.pkl", 'rb'))
    self_bleu = {}
    for line_num, sentence_list in sentence_map.items():
        diff_list = []
        for a_idx, a in enumerate(sentence_list):
            for b_idx, b in enumerate(sentence_list):
                if a_idx != b_idx:
                    sent_bleu_score = sacrebleu.sentence_bleu([a], [b]).score / 100.
                    diff_list.append(1 - sent_bleu_score)
        sentence_self_bleu = sum(diff_list) / (len(sentence_list) * (len(sentence_list) - 1))
        self_bleu[line_num] = sentence_self_bleu
    pkl.dump(self_bleu, open("mtdd_self_bleu.pkl", 'wb'))

def read_self_bleu():
    sentence_map  = pkl.load(open("sentence_map.pkl", 'rb'))
    self_bleu  = pkl.load(open("mtdd_self_bleu.pkl", 'rb'))
    # import pdb; pdb.set_trace()

    count = 0
    value_count = 0
    for k, v in self_bleu.items():
        value_count += v
        count += 1

    print(value_count, count, value_count / count)


# gen_lxb_bleu()
# lxb_self_bleu()
read_self_bleu()