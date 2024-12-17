from collections import Counter
import nltk
from nltk import pos_tag
import spacy
from spacy import displacy
import string
from utils import *
import re
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
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

def clean_tokens(tokens):
    # Remove errors in the dataset
    tokens = re.sub(r'\bhowmanyof\b', 'how many of', tokens)
    tokens = re.sub(r'\bmanylights\b', 'many lights', tokens)
    tokens = re.sub(r'\bJhow\b', 'how', tokens)
    tokens = re.sub(r'\bmanyof\b', 'many of', tokens)
    tokens = re.sub(r'\bclocks.are\b', 'clocks are', tokens)
    tokens = re.sub(r'\bt.v\b', 'tv', tokens)
    return tokens

def get_subject_freq_distribution(sents):
    # List of nouns that were incorrectly classified by spaCy
    noun_incorrectly_classified = ['utensils', 'kites', 'surfboard', 'covers', 'red stands', 'barefoot', 'geese', 'stems', 'squash', 
                                   'wristbands', 'mobile', 'giraffe', 'wakeboard', 'hydrant', 'elephant', 'sink', 'skateboarder', 'tub', 
                                   'salads', 'couches', 'parrot', 'remote', 'skateboard', 'shrimp', 'buffalo', 'sheep', 'wakeboard', 
                                   'building', 'burritos', 'salads', 'toothbrushe']
    # Dictionary to store the frequency of each class
    ret = {keys: {} for keys in sents.keys()}
    ret = dict(sorted(ret.items()))

    for key in sents.keys():
        dic = {}
        for sent in tqdm(sents[key], desc=f"Processing {key}"):
            # Remove errors in the dataset
            sent = clean_tokens(" ".join(sent))
            doc = nlp(sent)
            class_name = ""
            var = 0
            for token in doc:
                # Find "how many"
                if token.text.lower() == "how" and token.nbor(1).text.lower() == "many":
                    # Find the noun after "how many"
                    for possible_noun in doc[token.i+2:]:
                        if possible_noun.pos_ in ["NOUN", "PROPN"]:  # Noun or proper noun
                            # Extract the noun and its related tokens
                            related_tokens = [possible_noun]
                            for child in possible_noun.children:
                                if child.dep_ in ["amod", "det", "compound", "nmod"] and child.text.lower() not in ["many", "how"]:
                                    related_tokens.append(child)
                            # Reconstruct the class name
                            related_tokens.sort(key=lambda t: t.i)  # Sort by token index
                            class_name = " ".join([t.text for t in related_tokens])
                            break
                    # If no noun was found, check if the sentence contains a noun that was incorrectly classified
                    if class_name == "": 
                        for possible_noun in doc:
                            if possible_noun.text.lower() in noun_incorrectly_classified:
                                class_name = possible_noun.text.lower()
                                break
                    if class_name != "":
                        if "How" in class_name or "many" in class_name:
                            print(class_name)
                        if class_name not in dic:
                            dic[class_name] = 0
                        # Increment the class frequency
                        dic[class_name] += 1
                        break
        ret[key] = dic
    return ret