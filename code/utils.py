# -*- coding:utf-8 -*-
# author: Xiaoyu Xing & Zhijing Jin
# datetime: 2020/6/3

import os
import re
import json
import nltk
import random
import stanza
from copy import deepcopy
from nltk.corpus import wordnet as wn
from allennlp.predictors.predictor import Predictor

stanza.download('en')


class Utils():
    def __init__(self, data_folder='data/src_data/rest/'):
        self.negative_words_list = [
            'doesn\'t', 'don\'t', 'didn\'t', 'no', 'did not', 'do not',
            'does not', 'not yet', 'not', 'none', 'no one', 'nobody', 'nothing',
            'neither', 'nowhere', 'never', 'hardly', 'scarcely', 'barely'
        ]
        self.negative_words_list = sorted(self.negative_words_list,
                                          key=lambda s: len(s), reverse=True)
        self.degree_word_list = [
            'absolutely', 'awfully', 'badly', 'barely', 'completely',
            'decidedly', 'deeply', 'enormously', 'entirely', 'extremely',
            'fairly', 'fully',
            'greatly', 'highly',
            'incredibly', 'indeed', 'very', 'really'
        ]
        text = self.read_text(
            [os.path.join(data_folder, 'train_sent.json'),
             os.path.join(data_folder, 'dev_sent.json'),
             os.path.join(data_folder, 'test_sent.json'), ])

        self.word2idx = self.get_word2id(text)
        self.predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos',
                                   tokenize_pretokenized=True)

    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def untokenize(self, words):
        """
        Untokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `untokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.
        """
        text = ' '.join(words)
        step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',
                                                                     '...')
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ").replace(' - ',
                                                                        '-').replace(
            ' / ', '/')
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
            "can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        step7 = step6.replace("DELETE", "")
        step8 = re.sub(r"\s{2,}", " ", step7)
        return step8.strip()

    def tokenize_term_list(self, copy_sent, sent_example):
        term_to_position_list = {}
        term_list = sent_example['term_list']
        for tid in term_list:
            if tid not in term_to_position_list:
                term_to_position_list[tid] = {}
            opinion_to_position_list = []
            opinions = term_list[tid]['opinion_words']
            opinions_spans = term_list[tid]['opinion_position']
            polarity = term_list[tid]['polarity']
            for i in range(len(opinions)):
                posi = opinions_spans[i]
                opi_from = posi[0]
                opi_to = posi[1]
                left = self.tokenize(copy_sent[:opi_from].strip())
                opi = self.tokenize(copy_sent[opi_from:opi_to].strip())
                opinion_to_position_list.append(
                    [' '.join(opi), [len(left), len(left) + len(opi)]])

            fromidx = term_list[tid]['from']
            toidx = term_list[tid]['to']
            left = self.tokenize(copy_sent[:fromidx].strip())
            aspect = self.tokenize(copy_sent[fromidx:toidx].strip())
            term_to_position_list[tid]['id'] = tid
            term_to_position_list[tid]['term'] = term_list[tid]['term']
            term_to_position_list[tid]['from'] = len(left)
            term_to_position_list[tid]['to'] = len(left) + len(aspect)
            term_to_position_list[tid]['polarity'] = polarity
            term_to_position_list[tid]['opinions'] = opinion_to_position_list
        return term_to_position_list

    def reverse(self, words_list, opinions):
        new_words = deepcopy(words_list)
        new_opi_words = []
        from_to = []
        for i in range(len(opinions)):
            opi = opinions[i][0]
            opi_position = opinions[i][1]
            opi_from = opi_position[0]
            opi_to = opi_position[1]

            has_neg = False

            for w in self.negative_words_list:
                ws = self.tokenize(w)
                for j in range(opi_from, opi_to - len(ws) + 1):
                    new_words_ = ' '.join(new_words[j:j + len(ws)])
                    ws_ = ' '.join(ws)
                    if new_words_.lower() == ws_.lower():
                        if j > opi_from:
                            opi_to = opi_to - len(ws)
                            new_words[j: j + len(ws)] = ['DELETE'] * len(ws)
                            has_neg = True
                            break
                        else:
                            opi_from = j + len(ws)
                            new_words[j: j + len(ws)] = ['DELETE'] * len(ws)
                            has_neg = True
                            break
                if has_neg:
                    break
            opi_list = new_words[opi_from:opi_to]
            opi_tag, opi_tag_uni = self.get_postag(new_words, opi_from, opi_to)

            opi_words = new_words[opi_from:opi_to]
            if len(opi_list) == 1:
                opi = opi_list[0]

                if has_neg:
                    # delete negation words
                    if [opi_from, opi_to] not in from_to:
                        new_opi_words.append(
                            [opi_from, opi_to, self.untokenize(opi_words)])
                        from_to.append([opi_from, opi_to])
                else:
                    candidate = self.get_antonym_words(opi)
                    refined_candidate = self.refine_candidate(new_words,
                                                              opi_from, opi_to,
                                                              candidate)

                    if len(refined_candidate) == 0:
                        # negate the closest verb
                        opi_tag2, opi_tag_uni2 = self.get_postag(new_words, 0,
                                                                 -1)
                        if opi_tag_uni[0][-1] == 'ADJ' or opi_tag_uni[0][
                            -1] == 'NOUN' or opi_tag_uni[0][-1] == 'VERB':
                            if [opi_from, opi_to] not in from_to:
                                new_opi_words.append([opi_from, opi_to,
                                                      self.untokenize(
                                                          ['not', opi])])
                                from_to.append([opi_from, opi_to])
                        else:
                            dis = 1e10
                            fidx = -1
                            for idx, (w, t) in enumerate(opi_tag2):
                                if abs(idx - opi_from) < dis and w in ['is',
                                                                       'was',
                                                                       'are',
                                                                       'were',
                                                                       'am',
                                                                       'being']:
                                    dis = abs(idx - opi_from)
                                    fidx = idx
                            if fidx == -1:
                                if [opi_from, opi_to] not in from_to:
                                    new_opi_words.append([opi_from, opi_to,
                                                          self.untokenize(
                                                              ['not', opi])])
                                    from_to.append([opi_from, opi_to])
                            else:
                                if [opi_from, opi_to] not in from_to:
                                    new_opi_words.append(
                                        [fidx, fidx + 1, self.untokenize(
                                            [opi_tag_uni2[fidx][0], 'not'])])
                                    from_to.append([opi_from, opi_to])
                    else:
                        select = random.randint(0, len(refined_candidate) - 1)
                        if [opi_from, opi_to] not in from_to:
                            new_opi_words.append([opi_from, opi_to,
                                                  self.untokenize([
                                                      refined_candidate[
                                                          select]])])
                            from_to.append([opi_from, opi_to])
            elif len(opi_list) > 1:
                if has_neg:
                    new_opi_words.append(
                        [opi_from, opi_to, self.untokenize(opi_words)])
                else:
                    # negate the closest verb
                    new_opi_words.append(
                        [opi_from, opi_to, self.untokenize(
                            ['not ' + opi_words[0]] + opi_words[1:])])

        for nopi in new_opi_words:
            new_words[nopi[0]:nopi[1]] = [nopi[2]]
        return new_words, new_opi_words

    def exaggerate(self, words_list, opinions):
        new_words = deepcopy(words_list)
        new_opi_words = []
        for i in range(len(opinions)):
            opi_position = opinions[i][1]
            opi_from = opi_position[0]
            opi_to = opi_position[1]

            new_words = self.add_degree_words(new_words, opi_from, opi_to)
            new_opi_word = self.untokenize(new_words[opi_from:opi_to])
            new_opi_words.append([opi_from, opi_to, new_opi_word])

        return new_words, new_opi_words

    def get_postag(self, x, s, e):
        # TODO
        doc = self.nlp([x])
        # tags = nltk.pos_tag(x)
        # simple_tags = [(w, nltk.tag.map_tag('en-ptb', 'universal', tag)) for w, tag in tags]
        # if e != -1:
        #     return tags[s:e], simple_tags[s:e]
        # else:
        #     return tags[s:], simple_tags[s:]
        tags = [word.xpos for sent in doc.sentences for word in sent.words]
        simple_tags = [word.upos for sent in doc.sentences for word in
                       sent.words]
        words = [word.text for sent in doc.sentences for word in sent.words]

        t1 = list(zip(words, tags))
        t2 = list(zip(words, simple_tags))
        if e != -1:
            return t1[s:e], t2[s:e]
        else:
            return t1[s:], t2[s:]

    def get_antonym_words(self, word):
        antonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name())
        return antonyms

    def refine_candidate(self, words_list, opi_from, opi_to, candidate_list):
        if len(words_list) == 0:
            return []
        postag_list, _ = self.get_postag(words_list, 0, -1)
        postag_list = [t[1] for t in postag_list]
        in_vocab_candidate_list = []
        for candidate in candidate_list:
            if candidate.lower() in self.word2idx:
                in_vocab_candidate_list.append(candidate)

        refined_candi = []
        for candidate in in_vocab_candidate_list:
            opi = words_list[opi_from:opi_to][0]
            isupper = opi[0].isupper()
            allupper = opi.isupper()

            if allupper:
                candidate = candidate.upper()
            elif isupper:
                candidate = candidate[0].upper() + candidate[1:]
            if opi_from == 0:
                candidate = candidate[0].upper() + candidate[1:]

            new_words = words_list[:opi_from] + [candidate] + words_list[
                                                              opi_to:]

            # check pos tag
            new_postag_list, _ = self.get_postag(new_words, 0, -1)
            new_postag_list = [t[1] for t in new_postag_list]

            if len([i for i, j in zip(postag_list[opi_from:opi_to],
                                      new_postag_list[opi_from:opi_to]) if
                    i != j]) != 0:
                continue

            refined_candi.append(candidate)

        if len(refined_candi) == 0:
            for candidate in candidate_list:
                opi = words_list[opi_from:opi_to][0]
                isupper = opi[0].isupper()
                allupper = opi.isupper()

                if allupper:
                    candidate = candidate.upper()
                elif isupper:
                    candidate = candidate[0].upper() + candidate[1:]
                if opi_from == 0:
                    candidate = candidate[0].upper() + candidate[1:]

                new_words = words_list[:opi_from] + [candidate] + words_list[
                                                                  opi_to:]

                # check pos tag
                new_postag_list, _ = self.get_postag(new_words, 0, -1)
                new_postag_list = [t[1] for t in new_postag_list]

                if len([i for i, j in zip(postag_list[opi_from:opi_to],
                                          new_postag_list[opi_from:opi_to]) if
                        i != j]) != 0:
                    continue

                refined_candi.append(candidate)
        return refined_candi

    def get_word2id(self, text, lower=True):
        word2idx = {}
        idx = 1
        if lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
        return word2idx

    def read_text(self, fnames):
        text = ''
        for fname in fnames:
            with open(fname, 'r') as f:
                lines = json.load(f)
            for id in lines:
                instance = lines[id]
                text_instance = instance['sentence']
                # print(text_instance)
                text_raw = " ".join(self.process_text(text_instance)).lower()
                text += text_raw + " "
        return text.strip()

    def process_text(self, x):
        x = x.lower()
        x = x.replace("&quot;", " ")
        x = x.replace('"', " ")
        x = re.sub('[^A-Za-z0-9]+', ' ', x)
        x = x.strip().split(' ')
        # x = [strip_punctuation(y) for y in x]
        ans = []
        for y in x:
            if len(y) == 0:
                continue
            ans.append(y)
        # ptxt = nltk.word_tokenize(ptxt)
        return ans

    def add_degree_words(self, word_list, from_idx, to_idx):
        candidate_list = self.degree_word_list
        select = random.randint(0, len(candidate_list) - 1)
        opi = [' '.join([candidate_list[select]] + word_list[from_idx:to_idx])]
        new_words = word_list[:from_idx] + opi + word_list[to_idx:]
        return new_words

    def get_constituent(self, x):
        annotations = self.predictor.predict(sentence=x)['trees']
        return annotations

    def get_phrase(self, word, opi, ptree):
        phrase_level = [
            'ASJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX',
            'PP', 'PRN', 'PRT', 'QP', 'RRC',
            'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X', 'S', 'SBAR'
        ]
        phrase = []
        for node in ptree.subtrees(filter=lambda t: t.label() in phrase_level):
            if node.label() == 'NP':
                if node.right_sibling() != None and node.right_sibling().label() == 'VP':
                    continue
            if node.label() == 'VP':
                if node.left_sibling() != None and node.left_sibling().label() == 'NP':
                    continue
            if ''.join(word.split(' ')) in ''.join(node.leaves()) and ''.join(
                    opi.split(' ')) in ''.join(node.leaves()):
                phrase.append(node.leaves())
        phrase = sorted(phrase, key=len, reverse=True)
        return phrase
