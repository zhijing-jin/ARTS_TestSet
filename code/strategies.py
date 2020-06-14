# -*- coding:utf-8 -*-
# author: Xiaoyu Xing & Zhijing Jin
# datetime: 2020/6/3

import os
import json
import random
import string
import language_check
from nltk.tree import ParentedTree

from utils import Utils

random.seed(1016)


def revTgt(dataset, input_file, outfile):
    util = Utils(dataset)
    with open(input_file, 'r', encoding='utf-8') as fr:
        lines = json.load(fr)
    res = {}
    for sid in lines:
        sent_example = lines[sid]
        sentence = sent_example['sentence']
        copy_sent = sentence
        words_list = util.tokenize(copy_sent)
        term_to_position_list = util.tokenize_term_list(copy_sent, sent_example)
        for tid in term_to_position_list:
            term = term_to_position_list[tid]['term']
            term_from = term_to_position_list[tid]['from']
            term_to = term_to_position_list[tid]['to']
            polarity = term_to_position_list[tid]['polarity']
            opinions = term_to_position_list[tid]['opinions']
            other_polarity = set()
            other_opinions = set()
            for tid2 in term_to_position_list:
                if tid2 != tid:
                    other_polarity.add(term_to_position_list[tid2]['polarity'])
                    for other_opi in term_to_position_list[tid2]['opinions']:
                        other_opinions.add(other_opi[0])

            cur_opinions = set()
            for cur_opi in term_to_position_list[tid]['opinions']:
                cur_opinions.add(cur_opi[0])

            if polarity == 'positive':
                new_words, new_opi_words = util.reverse(words_list, opinions)
                new_polarity = 'negative'
            elif polarity == 'negative':
                new_words, new_opi_words = util.reverse(words_list, opinions)
                new_polarity = 'positive'
            else:
                new_words1, new_opi_words1 = util.reverse(words_list, opinions)
                new_words2, new_opi_words2 = util.reverse(words_list, opinions)
                if random.random() < 0.5:
                    new_words = new_words1
                    # new_opi_words = new_opi_words1
                else:
                    new_words = new_words2
                    # new_opi_words = new_opi_words2
                new_polarity = 'neutral'

            and_ind = []
            but_ind = []
            if len(other_polarity) > 0:
                for i, w in enumerate(new_words):
                    if w.lower() in ['and', 'but']:
                        if new_polarity not in other_polarity and w.lower() == 'and' and len(
                                cur_opinions & other_opinions) == 0 and w.lower() not in term.lower():
                            and_ind.append(i)
                        elif new_polarity in other_polarity and w.lower() == 'but' and len(
                                cur_opinions & other_opinions) == 0 and w.lower() not in term.lower():
                            but_ind.append(i)

            min = 1e8
            aidx = -1
            if new_polarity not in other_polarity and len(but_ind) == 0 and len(
                    cur_opinions & other_opinions) == 0:
                for idx in and_ind:
                    if idx > term_to and idx < len(new_words) - 3:
                        if idx - term_to < min:
                            min = idx - term_to
                            aidx = idx
                    if idx < term_from and idx > 3:
                        if term_to - idx < min:
                            min = term_to - idx
                            aidx = idx
                if aidx != -1:
                    new_words[aidx] = 'but'
            elif new_polarity in other_polarity and len(and_ind) == 0 and len(
                    cur_opinions & other_opinions) == 0:
                for idx in and_ind:
                    if idx > term_to and idx < len(new_words) - 3:
                        if idx - term_to < min:
                            min = idx - term_to
                            aidx = idx
                    if idx < term_from and idx > 3:
                        if term_to - idx < min:
                            min = term_to - idx
                            aidx = idx
                if aidx != -1:
                    new_words[aidx] = 'and'

            if new_words == words_list:
                continue

            new_sent = util.untokenize(new_words)
            if sentence[0].isupper():
                new_sent = new_sent[0].upper() + new_sent[1:]

            a = ''.join(util.tokenize(new_sent))
            b = ''.join(util.tokenize(term))
            c = 0
            for dd in range(len(a)):
                if a[dd:dd + len(b)] == b:
                    c = len(a[:dd])
                    break

            span_from = 0
            c2 = 0
            for dd in range(len(new_sent)):
                if new_sent[dd] != ' ':
                    c2 += 1
                if c2 == c and c != 0 and new_sent[dd + 1] != ' ':
                    span_from = dd + 1
                    break
                if c2 == c and c != 0 and new_sent[dd + 1] == ' ':
                    span_from = dd + 2
                    break

            span_to = span_from + len(term)

            if new_sent[span_from:span_to] != term:
                print(tid, term, new_sent[span_from:span_to])

            if sentence[0].isupper():
                new_sent = new_sent[0].upper() + new_sent[1:]

            print(new_sent)
            res[tid] = {
                'term': term,
                'id': tid,
                'sentence': new_sent,
                'multi': sent_example['multi'],
                'contra': sent_example['contra'],
                'from': span_from,
                'to': span_to,
                'polarity': new_polarity
            }
    with open(outfile, 'w', encoding='utf-8') as fw:
        json.dump(res, fw, indent=4)


def revNon(dataset, input_file, outfile):
    util = Utils(dataset)
    with open(input_file, 'r', encoding='utf-8') as fr:
        lines = json.load(fr)

    res = {}
    for sid in lines:
        sent_example = lines[sid]
        sentence = sent_example['sentence']
        term_list = sent_example['term_list']

        # do in multiple terms
        if len(term_list) == 1:
            continue

        copy_sent = sentence
        words_list = util.tokenize(copy_sent)
        term_to_position_list = util.tokenize_term_list(copy_sent, sent_example)

        all_id = []
        all_polarity = []
        all_opinions = []
        for tid in term_to_position_list:
            term = term_to_position_list[tid]['term']
            term_from = term_to_position_list[tid]['from']
            term_to = term_to_position_list[tid]['to']
            polarity = term_to_position_list[tid]['polarity']
            opinions = term_to_position_list[tid]['opinions']
            all_id.append(tid)
            all_polarity.append(polarity)
            all_opinions.append(opinions)

        for curid in range(len(all_id)):
            term = term_to_position_list[all_id[curid]]['term']
            term_from = term_to_position_list[all_id[curid]]['from']
            term_to = term_to_position_list[all_id[curid]]['to']
            cur_polarity = all_polarity[curid]
            other_id = all_id[:curid] + all_id[curid + 1:]
            cur_opinions = term_to_position_list[all_id[curid]]['opinions']
            cur_opinions_positions = [i[1] for i in cur_opinions]

            change_words = words_list
            change_opi_words = []
            change_opinion_position = []

            for ix, id in enumerate(other_id):
                other_opinions = term_to_position_list[id]['opinions']
                other_opinions_positions = [i[1] for i in other_opinions]

                find = False
                for i in other_opinions_positions:
                    if i in cur_opinions_positions:
                        find = True
                        break
                if find:
                    continue

                non_overlap_opinions = []
                for op in other_opinions:
                    if op[1] in change_opinion_position:
                        continue
                    else:
                        non_overlap_opinions.append(op)

                if len(non_overlap_opinions) == 0:
                    continue

                if cur_polarity == term_list[other_id[ix]]['polarity']:
                    if term_list[other_id[ix]]['polarity'] == 'positive':
                        new_words, new_opi_words = util.reverse(change_words,
                                                                non_overlap_opinions)
                    elif term_list[other_id[ix]]['polarity'] == 'negative':
                        new_words, new_opi_words = util.reverse(change_words,
                                                                non_overlap_opinions)
                    else:
                        continue

                    and_ind = []
                    for i, w in enumerate(new_words):
                        if w.lower() in [
                            'and'] and w.lower() not in term.lower():
                            and_ind.append(i)

                    min = 1e8
                    aidx = -1
                    for idx in and_ind:
                        if idx > term_to and idx < len(new_words) - 3:
                            if idx - term_to < min:
                                min = idx - term_to
                                aidx = idx
                        if idx < term_from and idx > 3:
                            if term_to - idx < min:
                                min = term_to - idx
                                aidx = idx
                    if aidx != -1:
                        new_words[aidx] = 'but'
                else:
                    new_words, new_opi_words = util.exaggerate(change_words,
                                                               non_overlap_opinions)

                if new_words != change_words:
                    for op in new_opi_words:
                        change_opinion_position.append([op[0], op[1]])
                    change_words = new_words

            if change_words == words_list:
                continue

            new_sent = util.untokenize(change_words)

            a = ''.join(util.tokenize(new_sent))
            b = ''.join(util.tokenize(term))
            c = 0
            for i in range(len(a)):
                if a[i:i + len(b)] == b:
                    c = len(a[:i])
                    break

            span_from = 0
            c2 = 0
            for i in range(len(new_sent) - 1):
                if new_sent[i] != ' ':
                    c2 += 1
                if c2 == c and c != 0 and new_sent[i + 1] != ' ':
                    span_from = i + 1
                    break
                if c2 == c and c != 0 and new_sent[i + 1] == ' ':
                    span_from = i + 2
                    break

            span_to = span_from + len(term)

            if new_sent[span_from:span_to] != term:
                print(all_id[curid], term, new_sent[span_from:span_to])

            print(new_sent)

            if sentence[0].isupper():
                new_sent = new_sent[0].upper() + new_sent[1:]

            res[all_id[curid]] = {
                'term': term,
                'id': all_id[curid],
                'sentence': new_sent,
                'multi': sent_example['multi'],
                'contra': sent_example['contra'],
                'from': span_from,
                'to': span_to,
                'polarity': cur_polarity
            }

    with open(outfile, 'w', encoding='utf-8') as fw:
        json.dump(res, fw, indent=4)
    print(len(res))


def addDiff(dataset, infile, infile2, outfile, same=False):
    tool = language_check.LanguageTool('en-US')
    util = Utils(dataset)
    with open(infile, 'r', encoding='utf-8') as fw:
        examples = json.load(fw)

    pos_noun_adj_pair = []
    neg_noun_adj_pair = []
    neu_noun_adj_pair = []
    for id in examples:
        example = examples[id]
        term_list = example['term_list']
        sentence = example['sentence']
        annotations = util.get_constituent(sentence)

        for i, tid in enumerate(term_list):
            term = term_list[tid]['term']
            opinion = term_list[tid]['opinion_words'][-1].lower()
            try:
                # some sentence can not be parsed
                ptree = ParentedTree.fromstring(annotations)
            except:
                continue
            # ancestor = util.get_ancestor(ptree, term, opinion)
            # opi_words = ancestor.leaves()
            phrases = util.get_phrase(term, opinion, ptree)

            other_terms = []
            for otid in term_list:
                if otid != tid:
                    other_terms.append(
                        ''.join(term_list[otid]['term'].split(' ')))

            opi_words = []
            for p in phrases:
                p_ = ''.join(p)
                overlap = False
                for other_term in other_terms:
                    if other_term in p_:
                        overlap = True
                        break
                if not overlap:
                    opi_words.append(p)

            opi_words = sorted(opi_words, key=len)

            if len(opi_words) == 0:
                continue
            opi_words = opi_words[0]

            if term_list[tid]['polarity'] == 'positive':
                pos_noun_adj_pair.append(
                    (term.lower(), [k.lower() for k in opi_words]))
            elif term_list[tid]['polarity'] == 'negative':
                neg_noun_adj_pair.append(
                    (term.lower(), [k.lower() for k in opi_words]))
            elif term_list[tid]['polarity'] == 'neutral':
                neu_noun_adj_pair.append(
                    (term.lower(), [k.lower() for k in opi_words]))

    with open(infile2, 'r', encoding='utf-8') as fw:
        examples = json.load(fw)

    res = {}
    for id in examples:
        sent_example = examples[id]
        term_list = sent_example['term_list']
        all_term = []
        for tid in term_list:
            all_term.append(term_list[tid]['term'])

        for tid in term_list:
            sentence = sent_example['sentence']

            fromidx = term_list[tid]['from']
            toidx = term_list[tid]['to']

            term = term_list[tid]['term']
            polarity = term_list[tid]['polarity']
            if same:
                if polarity == 'positive':
                    pair = pos_noun_adj_pair
                elif polarity == 'negative':
                    pair = neg_noun_adj_pair
                else:
                    pair = neu_noun_adj_pair
            else:
                if polarity == 'positive':
                    pair = neg_noun_adj_pair
                elif polarity == 'negative':
                    pair = pos_noun_adj_pair
                else:
                    pair = neu_noun_adj_pair

            punct = '.'
            if sentence[-1] == string.punctuation:
                punct = sentence[-1]

            polarity_list = []
            while True:
                add_num = random.randint(1, 3)
                random_num1, random_num2, random_num3 = random.sample(
                    range(len(pair)), 3)
                random_pair1 = pair[random_num1]
                random_pair2 = pair[random_num2]
                random_pair3 = pair[random_num3]
                if random_pair1[0] not in all_term and random_pair2[
                    0] not in all_term and random_pair3[
                    0] not in all_term:
                    if random_pair1 in pos_noun_adj_pair:
                        polarity_list.append('positive')
                    elif random_pair1 in neg_noun_adj_pair:
                        polarity_list.append('negative')
                    else:
                        polarity_list.append('')

                    if random_pair2 in pos_noun_adj_pair:
                        polarity_list.append('positive')
                    elif random_pair2 in neg_noun_adj_pair:
                        polarity_list.append('negative')
                    else:
                        polarity_list.append('')

                    if random_pair3 in pos_noun_adj_pair:
                        polarity_list.append('positive')
                    elif random_pair3 in neg_noun_adj_pair:
                        polarity_list.append('negative')
                    else:
                        polarity_list.append('')
                    break

            polarity_dict = {'positive': 0, 'negative': 0}
            for tid_ in term_list:
                if term_list[tid_]['polarity'] in ['positive', 'negative']:
                    polarity_dict[term_list[tid_]['polarity']] += 1

            if add_num == 3:
                tmp_words = random_pair1[1] + [','] + random_pair2[1] + [
                    'and'] + random_pair3[1] + [punct]
                for m in polarity_list[:3]:
                    if len(m) > 0:
                        polarity_dict[m] += 1
            elif add_num == 2:
                tmp_words = random_pair1[1] + ['and'] + random_pair2[1] + [
                    punct]
                for m in polarity_list[:2]:
                    if len(m) > 0:
                        polarity_dict[m] += 1
            else:
                tmp_words = random_pair1[1] + [punct]
                for m in polarity_list[:1]:
                    if len(m) > 0:
                        polarity_dict[m] += 1

            while sentence[-1] in '.?!':
                sentence = sentence[:-1]

            opi_tag, opi_tag_uni = util.get_postag(tmp_words, 0, 1)

            if opi_tag_uni[0] != 'CONJ':
                tmp_sentence = 'but ' + util.untokenize(tmp_words)
                matches = tool.check(tmp_sentence)
                new_sentence = language_check.correct(tmp_sentence, matches)
                new_sentence = new_sentence[4:]

                if 'but' in sentence or 'although' in sentence:
                    new_sent = sentence + "; " + new_sentence
                else:
                    new_sent = sentence + ", but " + new_sentence
            else:
                tmp_sentence = util.untokenize(tmp_words)
                matches = tool.check(tmp_sentence)
                new_sentence = language_check.correct(tmp_sentence, matches)
                new_sent = sentence + ". " + new_sentence[
                    0].upper() + new_sentence[1:]

            # if new_sent[fromidx:toidx] != term:
            #     print("****not equal****")

            print(new_sent)
            if new_sent:
                res[tid] = {
                    'term': term,
                    'id': tid,
                    'sentence': new_sent,
                    'multi': sent_example['multi'],
                    'contra': sent_example['contra'],
                    'from': fromidx,
                    'to': fromidx + len(term),
                    'polarity': polarity,
                    'portion': polarity_dict
                }

    with open(outfile, 'w', encoding='utf-8') as fw:
        json.dump(res, fw, indent=4)


if __name__ == '__main__':
    # test method
    dataset = "laptop"
    strategy = 'revTgt'

    data_folder = 'data/src_data/{}/'.format(dataset)
    input_file = os.path.join(data_folder, 'test_sent_towe.json')
    output_file = os.path.join(data_folder, 'test_adv.json')

    if strategy == 'revTgt':
        revTgt(data_folder, input_file, output_file)
    elif strategy == 'revNon':
        revNon(data_folder, input_file, output_file)
    elif strategy == 'addDiff':
        addDiff(dataset, os.path.join(data_folder, 'train_sent_towe.json'),
                os.path.join(data_folder, 'test_sent.json'), output_file)
