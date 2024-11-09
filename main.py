import os
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Importing the functions from the files
from utils import *
from stats import * 


DATASET = ["train", "test_complex", "test_simple"]
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, "total"]
TOP = 10 # Number of top frequent words to display in the chart
POS = ["NN", "VB"]


if __name__ == "__main__":
    #* Download the nltk resources (if not already downloaded)
    #nltk.download('punkt_tab')
    #nltk.download('stopwords')
    #nltk.download('averaged_perceptron_tagger_eng')

    # Load the data from the json files
    train_raw = load_data(os.path.join('dataset','tallyqa','train.json'))
    test_raw = load_data(os.path.join('dataset','tallyqa','test.json'))

    # Get the sentences divided by class of the training and test data
    sents_train = get_sents(train_raw, divided_by_class=True)
    sents_test_complex, sents_test_simple = get_sents(test_raw, divided_by_class=True)
    
    # Group all the sentences
    all_sents_train = group_sents(sents_train)
    all_sents_test_complex = group_sents(sents_test_complex)
    all_sents_test_simple = group_sents(sents_test_simple)

    #* min, max and average sentence length
    print(stats_sent_len(all_sents_train))
    print(stats_sent_len(all_sents_test_complex))
    print(stats_sent_len(all_sents_test_simple))

    #* Number of sentences in the training and test set
    print(f"Number of sentences in the training set: {len(all_sents_train['all_classes'])}")
    print(f"Number of sentences in the test set (complex): {len(all_sents_test_complex['all_classes'])}")
    print(f"Number of sentences in the test set (simple): {len(all_sents_test_simple['all_classes'])}")

    # Get the frequency distribution of the words in the training set and test set
    freq_dist_classes_train = get_freq_dist(sents_train, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)
    freq_dist_total_train = get_freq_dist(all_sents_train, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)

    freq_dist_classes_test_complex = get_freq_dist(sents_test_complex, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)
    freq_dist_total_test_complex = get_freq_dist(all_sents_test_complex, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)

    freq_dist_classes_test_simple = get_freq_dist(sents_test_simple, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)
    freq_dist_total_test_simple = get_freq_dist(all_sents_test_simple, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)
    
    #* Vocalurary size of the training and test set
    print(vocabulary_lengths(freq_dist_total_train))
    print(vocabulary_lengths(freq_dist_total_test_complex))
    print(vocabulary_lengths(freq_dist_total_test_simple))

    #* Plot the frequency distribution of the words in the training set and test set
    '''
    # Map dataset names to their corresponding dictionaries
    freq_dist_classes = {
        "train": freq_dist_classes_train,
        "test_complex": freq_dist_classes_test_complex,
        "test_simple": freq_dist_classes_test_simple,
    }

    freq_dist_totals = {
        "train": freq_dist_total_train,
        "test_complex": freq_dist_total_test_complex,
        "test_simple": freq_dist_total_test_simple,
    }

    for elem in DATASET:
        for c in CLASSES:
            path = f"results/freq_dist/{elem}/top_{TOP}_most_freq_words_in_{c}.png"
            
            # Pass the appropriate dictionaries for each dataset
            plot_word_frequencies(
                class_dicts=freq_dist_classes[elem],
                all_classes_freq=freq_dist_totals[elem]["all_classes"],
                top_n=TOP,
                mode=c,
                save_path=path
            )
    
    '''
    #* Histogram  of answer distribution
    '''
    path = "results/histograms_answer_counts/train"
    plot_histogram(sents_train,title="Answer distribution in the training set", output_folder=path)
    path = "results/histograms_answer_counts/test_complex"
    plot_histogram(sents_test_complex,title="Answer distribution in the test set (complex)", output_folder=path)
    path = "results/histograms_answer_counts/test_simple"
    plot_histogram(sents_test_simple,title="Answer distribution in the test set (simple)", output_folder=path)
    '''

    #* POS tagging
    '''
    pos_fd_classes_train = pos_distribution(freq_dist_classes_train)
    pos_fd_total_train = pos_distribution(freq_dist_total_train)
    pos_fd_classes_test_complex = pos_distribution(freq_dist_classes_test_complex)
    pos_fd_total_test_complex = pos_distribution(freq_dist_total_test_complex)
    pos_fd_classes_test_simple = pos_distribution(freq_dist_classes_test_simple)
    pos_fd_total_test_simple = pos_distribution(freq_dist_total_test_simple)

    pos_fd_classes = {
        "train": pos_fd_classes_train,
        "test_complex": pos_fd_classes_test_complex,
        "test_simple": pos_fd_classes_test_simple,
    }
    pos_fd_totals = {
        "train": pos_fd_total_train,
        "test_complex": pos_fd_total_test_complex,
        "test_simple": pos_fd_total_test_simple,
    }

    #* Plot the POS distribution
    for p in POS:
        for elem in DATASET:
            for c in CLASSES:
                output_folder = f"results/freq_dist_pos/{p}/{elem}"
                os.makedirs(output_folder, exist_ok=True)
                path = output_folder + f"/top_{TOP}_most_freq_{p}_in_{c}.png"
                plot_word_frequencies(
                    class_dicts=filter_freq_dist(pos_fd_classes[elem], p),
                    all_classes_freq=filter_freq_dist(pos_fd_totals[elem], p)["all_classes"],
                    top_n=TOP,
                    mode=c,
                    save_path=path,
                    title=f"Top {TOP} Most Frequent {p} in {c}"
                )
    '''