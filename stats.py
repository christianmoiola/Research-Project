from collections import Counter
import nltk
from nltk import pos_tag
from utils import *
import string

STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def get_sents(data, divided_by_class=False):
    '''
    Function to get the sentences from the data, divided by class if specified.
    '''
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
    '''
    Function to get the minimum, maximum and average sentence length of the sentences.
    '''
    ret = {keys: {} for keys in sents.keys()}
    for key in sents.keys():
        sents_len = [len(sent) for sent in sents[key]]

        avg_sent_len = round(sum(sents_len) / len(sents_len))
        min_sent_len = min(sents_len)
        max_sent_len = max(sents_len)
        ret[key] = {"avg_sent_len": avg_sent_len, "min_sent_len": min_sent_len, "max_sent_len": max_sent_len}
    return ret

def get_freq_dist(sents, lower=True, remove_stopwords=True, freq_cutoff=None, remove_punctuation=True):
    '''
    Function to get the frequency distribution of the words in the sentences.

    Parameters:
    - sents (dict): Dictionary with keys from 0 to 15 or all_classes, each containing a list of sentences.
    - lower (bool): Whether to convert all words to lowercase.
    - remove_stopwords (bool): Whether to remove stopwords from the frequency distribution.
    - freq_cutoff (int, optional): Minimum frequency for a word to be included in the distribution.
    - remove_punctuation (bool): Whether to remove punctuation (e.g., '.', ',', '?') from the frequency distribution.
    
    Output:
    - freq_dist (dict): Dictionary with keys from 0 to 15 or all_classes, each containing the frequency distribution of words.
    '''
    ret = {keys: {} for keys in sents.keys()}
    ret = dict(sorted(ret.items()))
    
    for key in sents.keys():
        words = flatten_list(sents[key])
        if lower:
            words = [word.lower() for word in words]
        if remove_stopwords:
            words = [word for word in words if word not in STOPWORDS]
        if remove_punctuation:
            words = [word for word in words if word not in string.punctuation and word not in ["``", "''"]]
        
        freq_dist = nltk.FreqDist(words)
        if freq_cutoff:
            freq_dist = {k: v for k, v in freq_dist.items() if v >= freq_cutoff}
        ret[key] = freq_dist

    return ret

def pos_distribution(dictionary):
    '''
    For each class, create a dictionary with POS categories as keys and word frequencies as items.
    '''
    result = {}
    
    for class_name, vocab_dict in dictionary.items():
        # Dictionary to store the POS categories and their word frequencies
        pos_dict = {}
        
        for word, frequency in vocab_dict.items():
            # Get the POS tag for the word
            tagged = pos_tag([word])
            
            token, tag = tagged[0]
            # Add the word frequency to the corresponding POS category
            if tag not in pos_dict:
                pos_dict[tag] = {}
            if token not in pos_dict[tag]:
                pos_dict[tag][token] = 0
            pos_dict[tag][token] += frequency
        
        result[class_name] = pos_dict
    
    return result

