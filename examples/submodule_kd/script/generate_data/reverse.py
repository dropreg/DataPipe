


with open("/data/lxb/sm_gen_data/de-en-data/reverse/de-en.en_0", 'w') as fw:
    for line in open("/data/lxb/sm_gen_data/de-en-data/reverse/de-en.en").readlines():
        word_list = line.strip().split()
        fw.writelines(" ".join(word_list[::-1]) + '\n')
