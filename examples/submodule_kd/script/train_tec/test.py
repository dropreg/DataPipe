



for i in range(6):
    count = 0
    with open("/home/lxb/code/fairseq/en.pred{}".format(i), 'w') as fw:
        for lines in open("/home/lxb/code/fairseq/en.pred").readlines():
            if count % 6 == i:
                fw.writelines(lines)
            count += 1
