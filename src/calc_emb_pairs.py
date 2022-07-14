import sys
import pickle
import numpy as np


def run(model_num):
    dcts2 = pickle.load(open('our_dcts%d__271_0.25.pkl' % (7 + model_num), 'rb'))
    #
    padding = 0.06
    shift_img = 0.05
    embs_name = 'embs_' + ('AddModel_%d' % model_num) + '_%.2f_%.2f' % (padding, shift_img)

    ratios_dct = {}
    old_ratio = 0

    all_same_embs_pairs = []
    all_othr_embs_pairs = []
    # same
    for dct in dcts2:
        for i in range(len(dct[embs_name]) - 1):
            emb = dct[embs_name][i]
            for j in range(i + 1, len(dct[embs_name])):
                emb_ = dct[embs_name][j]
                d = emb - emb_
                d = np.sqrt(np.sum(d * d))
                all_same_embs_pairs.append(d)
    # others
    n_void = 0
    i_dc = 0
    j_dc = 0
    for dct in dcts2:
        for dct_ in dcts2:
            if i_dc == j_dc:
                j_dc += 1
                continue
            for i in range(len(dct[embs_name]) - 1):
                emb = dct[embs_name][i]
                for j in range(len(dct_[embs_name])):
                    if n_void % 150 == 0:
                        emb_ = dct_[embs_name][j]
                        d = emb - emb_
                        d = np.sqrt(np.sum(d * d))
                        all_othr_embs_pairs.append(d)
                    n_void += 1
            j_dc += 1
        j_dc = 0
        i_dc += 1

    pickle.dump((all_same_embs_pairs, all_othr_embs_pairs), open('embs_pairs_AddModel_%d.pkl' % model_num, 'wb'))
    print('embs_pairs_AddModel_%d.pkl saved' % model_num)


def main():
    if len(sys.argv) != 2:
        print("""USAGE: calc_emb_pairs.py model_num""")
        exit(1)

    run(int(sys.argv[1]))

################################################################################

if __name__ == "__main__":
    main()

################################################################################
