from collections import Counter
import nltk

from utils import *

STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def get_sents(data, divided_by_class=False):

    if divided_by_class:
        ret = {} if "issimple" not in data[0] else [{}, {}]
    else:
        ret = [] if "issimple" not in data[0] else [[], []]

    for elem in data:
        question = nltk.word_tokenize(elem["question"])

        if "issimple" not in elem:
            if divided_by_class:
                if elem["answer"] not in ret:
                    ret[elem["answer"]] = []
                ret[elem["answer"]].append(question)
            else:
                ret.append(question)

        else:
            index = 1 if elem["issimple"] else 0
            if divided_by_class:
                if elem["answer"] not in ret[index]:
                    ret[index][elem["answer"]] = []
                ret[index][elem["answer"]].append(question)
            else:
                ret[index].append(question)

    if divided_by_class == False:
        if "issimple" not in data[0]:
            ret = {"all_classes": ret}
        else:
            ret = [{"all_classes": ret[0]}, {"all_classes": ret[1]}]

    return ret

def stats_sent_len(sents):
    ret = {keys: {} for keys in sents.keys()}
    for key in sents.keys():
        sents_len = [len(sent) for sent in sents[key]]

        avg_sent_len = round(sum(sents_len) / len(sents_len))
        min_sent_len = min(sents_len)
        max_sent_len = max(sents_len)
        ret[key] = {"avg_sent_len": avg_sent_len, "min_sent_len": min_sent_len, "max_sent_len": max_sent_len}
    return ret

def get_freq_dist(sents, lower=True, remove_stopwords=True, freq_cutoff=None):

    ret = {keys: {} for keys in sents.keys()}
    ret = dict(sorted(ret.items()))
    for key in sents.keys():
        words = flatten_list(sents[key])
        if lower:
            words = [word.lower() for word in words]
        if remove_stopwords:
            words = [word for word in words if word not in STOPWORDS]
        
        freq_dist= nltk.FreqDist(words)
        if freq_cutoff:
            freq_dist = {k: v for k, v in freq_dist.items() if v >= freq_cutoff}
        ret[key] = freq_dist

    return ret

