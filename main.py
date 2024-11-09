import os
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Importing the functions from the files
from utils import *
from stats import * 


if __name__ == "__main__":
    # Download the nltk resources (if not already downloaded)
    #nltk.download('punkt_tab')
    #nltk.download('stopwords')

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

    # Min, max and average sentence length
    print(stats_sent_len(all_sents_train))
    print(stats_sent_len(all_sents_test_complex))
    print(stats_sent_len(all_sents_test_simple))

    # Number of sentences in the training and test set
    print(f"Number of sentences in the training set: {len(all_sents_train['all_classes'])}")
    print(f"Number of sentences in the test set (complex): {len(all_sents_test_complex['all_classes'])}")
    print(f"Number of sentences in the test set (simple): {len(all_sents_test_simple['all_classes'])}")

    # Get the frequency distribution of the words in the training set and test set
    freq_dist_classes_train = get_freq_dist(sents_train, lower=True, remove_stopwords=True, freq_cutoff=2)
    freq_dist_total_train = get_freq_dist(all_sents_train, lower=True, remove_stopwords=True, freq_cutoff=2)

    freq_dist_classes_test_complex = get_freq_dist(sents_test_complex, lower=True, remove_stopwords=True, freq_cutoff=2)
    freq_dist_total_test_complex = get_freq_dist(all_sents_test_complex, lower=True, remove_stopwords=True, freq_cutoff=2)

    freq_dist_classes_test_simple = get_freq_dist(sents_test_simple, lower=True, remove_stopwords=True, freq_cutoff=2)
    freq_dist_total_test_simple = get_freq_dist(all_sents_test_simple, lower=True, remove_stopwords=True, freq_cutoff=2)
    
    # Plot the frequency distribution of the words in the training set and test set
    '''
    dataset = ["train", "test_complex", "test_simple"]
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, "total"]
    top = 10 # Number of top frequent words to display in the chart
    
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

    for elem in dataset:
        for c in classes:
            path = f"results/freq_dist/{elem}/top_{top}_most_freq_words_in_{c}.png"
            
            # Pass the appropriate dictionaries for each dataset
            plot_word_frequencies(
                class_dicts=freq_dist_classes[elem],
                all_classes_freq=freq_dist_totals[elem]["all_classes"],
                top_n=top,
                mode=c,
                save_path=path
            )
    '''

    #freq_dist_complete = get_freq_dist(all_sents, lower=True, remove_stopwords=True, freq_cutoff=2)

    #plot_freq_dist(freq_dist_classes[1], top_n=10, title="Frequency distribution of the words in class 1")
    #plot_freq_dist(freq_dist_complete["all_classes"], top_n=10, title="Frequency distribution of the words in the complete dataset")



    

    #print(len(tmp_train_raw), len(test_raw))
    #print(tmp_train_raw[0])
    #data_source = [x['data_source'] for x in tmp_train_raw]
    #print(Counter(data_source))
    #data_source = [x['data_source'] for x in test_raw]
    #print(Counter(data_source))
    #answer = [x['answer'] for x in tmp_train_raw]

    #print(Counter(answer))

    #  Get the statistics of the data
    # 1. minimum/maximum/average number of character per token
    # 2. minimum/maximum/average number of words per sentence
    # 3. frequency of each word in the dataset (also with frequency cutoff and stop words removed)
    # 4. Analysis of the dictionary of the dataset

    # 1.
    #print(Counter(get_words(tmp_train_raw)))
    '''
    sents = get_sents(tmp_train_raw)
    words = flatten_list(sents)
    words_lower = [word.lower() for word in words]
    words_lower_filtered = [word for word in words_lower if word not in stopwords]

    #Stats of the sentences
    print(stats_sent_len(sents))

    lexicon = set(words)
    lexicon_lower = set(words_lower)
    lexicon_lower_filtered = set(words_lower_filtered)
    print(len(lexicon_lower_filtered.intersection(stopwords)))
    print("Lexicon size: ", len(lexicon), "Lexicon size (lower): ", len(lexicon_lower), "Total words: ", len(words))
    
    freq_dist = nltk.FreqDist(words_lower)
    freq_dist_filtered = nltk.FreqDist(words_lower_filtered)

    print(len(freq_dist.items()), len(freq_dist_filtered.items()))

    #Calcolare le parole più frequenti per ogni classe e confrontarle con le parole più frequenti in assoluto
    
    print(len(lexicon_lower.intersection(stopwords)))
    '''