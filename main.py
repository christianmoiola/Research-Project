import os
import nltk
from nltk.corpus import stopwords


from utils import *
from stats import *

class DatasetConfig:
    SPLITS = ["train", "test_complex", "test_simple"]
    CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, "total"]
    TOP_N_WORDS = 10
    POS_TAGS = ["NN", "VB"]

class DataAnalyzer:
    def __init__(self):
        self.config = DatasetConfig()
        self.data_stats = []
    
    def download_nltk_resources(self):
        """Download the nltk resources if not already downloaded"""
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger_eng')

    def load_datasets(self):
        """Load and process all datasets"""
        train_raw = load_data(os.path.join('dataset', 'tallyqa', 'train.json'))
        test_raw = load_data(os.path.join('dataset', 'tallyqa', 'test.json'))

        # Process training data
        self.sents_train = get_sents(train_raw, divided_by_class=True)
        self.all_sents_train = group_sents(self.sents_train)

        # Process test data
        self.sents_test_complex, self.sents_test_simple = get_sents(test_raw, divided_by_class=True)
        self.all_sents_test_complex = group_sents(self.sents_test_complex)
        self.all_sents_test_simple = group_sents(self.sents_test_simple)

        print(f"Number of sentences in the training set: {len(self.all_sents_train['all_classes'])}")
        print(f"Number of sentences in the test set (complex): {len(self.all_sents_test_complex['all_classes'])}")
        print(f"Number of sentences in the test set (simple): {len(self.all_sents_test_simple['all_classes'])}")

    def calculate_sentence_statistics(self):
        """Calculate statistics for all datasets"""
        datasets = [
            ("Train", self.all_sents_train),
            ("Test-Complex", self.all_sents_test_complex),
            ("Test-Simple", self.all_sents_test_simple)
        ]

        for name, dataset in datasets:
            stats = stats_sent_len(dataset)
            self.data_stats.append([
                name,
                len(dataset['all_classes']),
                stats["all_classes"]["min_sent_len"],
                stats["all_classes"]["max_sent_len"],
                stats["all_classes"]["avg_sent_len"],
                stats["all_classes"]["std_dev"]
            ])

    def calculate_frequency_distributions(self):
        """Calculate word frequency distributions for all datasets"""
        # Training set
        self.freq_dist_classes_train = get_freq_dist(self.sents_train, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)
        self.freq_dist_total_train = get_freq_dist(self.all_sents_train, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)

        # Test complex set
        self.freq_dist_classes_test_complex = get_freq_dist(self.sents_test_complex, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)
        self.freq_dist_total_test_complex = get_freq_dist(self.all_sents_test_complex, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)

        # Test simple set
        self.freq_dist_classes_test_simple = get_freq_dist(self.sents_test_simple, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)
        self.freq_dist_total_test_simple = get_freq_dist(self.all_sents_test_simple, lower=True, remove_stopwords=True, freq_cutoff=2, remove_punctuation=True)

        print(vocabulary_lengths(self.freq_dist_total_train))
        print(vocabulary_lengths(self.freq_dist_total_test_complex))
        print(vocabulary_lengths(self.freq_dist_total_test_simple))

    def calculate_subject_distributions(self):
        """Calculate subject frequency distributions for all datasets"""
        # Calculate distributions
        self.freq_subj_dist_classes_train = get_subject_freq_distribution(self.sents_train)
        self.freq_subj_dist_classes_test_complex = get_subject_freq_distribution(self.sents_test_complex)
        self.freq_subj_dist_classes_test_simple = get_subject_freq_distribution(self.sents_test_simple)

        # Flatten distributions
        self.freq_subj_dist_total_train = flatten_freq_distribution(self.freq_subj_dist_classes_train)
        self.freq_subj_dist_total_test_complex = flatten_freq_distribution(self.freq_subj_dist_classes_test_complex)
        self.freq_subj_dist_total_test_simple = flatten_freq_distribution(self.freq_subj_dist_classes_test_simple)

    def add_vocabulary_sizes(self):
        """Add vocabulary sizes to statistics"""
        vocab_sizes = [
            len(self.freq_dist_total_train['all_classes']),
            len(self.freq_dist_total_test_complex['all_classes']),
            len(self.freq_dist_total_test_simple['all_classes'])
        ]
        
        for i, vocab_size in enumerate(vocab_sizes):
            self.data_stats[i].append(vocab_size)

    def generate_results(self):
        """Generate all plots and visualizations"""
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Create statistics table
        headers = [
            "Split", "Number of sentences", "Min sentence length",
            "Max sentence length", "Average sentence length",
            "Standard deviation", "Vocabulary size"
        ]
        create_table(self.data_stats, headers, "results/table.png")

        # Plot word frequencies
        self._plot_word_frequencies()
        
        # Plot subject frequencies
        self._plot_subject_frequencies()
        
        # Plot answer distributions
        self._plot_answer_distributions()

    def _plot_word_frequencies(self):
        """Helper method to plot word frequencies"""
        freq_dist_classes = {
            "train": self.freq_dist_classes_train,
            "test_complex": self.freq_dist_classes_test_complex,
            "test_simple": self.freq_dist_classes_test_simple
        }

        freq_dist_totals = {
            "train": self.freq_dist_total_train,
            "test_complex": self.freq_dist_total_test_complex,
            "test_simple": self.freq_dist_total_test_simple
        }

        for dataset in self.config.SPLITS:
            for class_id in self.config.CLASSES:
                path = f"results/freq_dist/{dataset}/top_{self.config.TOP_N_WORDS}_most_freq_words_in_{class_id}.png"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                plot_word_frequencies(
                    class_dicts=freq_dist_classes[dataset],
                    all_classes_freq=freq_dist_totals[dataset]["all_classes"],
                    top_n=self.config.TOP_N_WORDS,
                    mode=class_id,
                    save_path=path
                )

    def _plot_subject_frequencies(self):
        """Helper method to plot subject frequencies"""
        freq_dist_classes = {
            "train": self.freq_subj_dist_classes_train,
            "test_complex": self.freq_subj_dist_classes_test_complex,
            "test_simple": self.freq_subj_dist_classes_test_simple
        }

        freq_dist_totals = {
            "train": self.freq_subj_dist_total_train,
            "test_complex": self.freq_subj_dist_total_test_complex,
            "test_simple": self.freq_subj_dist_total_test_simple
        }

        for dataset in self.config.SPLITS:
            for class_id in self.config.CLASSES:
                path = f"results/freq_subj_dist/{dataset}/top_{self.config.TOP_N_WORDS}_most_freq_subject_in_{class_id}.png"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                plot_word_frequencies(
                    class_dicts=freq_dist_classes[dataset],
                    all_classes_freq=freq_dist_totals[dataset]["all_classes"],
                    top_n=self.config.TOP_N_WORDS,
                    mode=class_id,
                    save_path=path
                )

    def _plot_answer_distributions(self):
        """Helper method to plot answer distributions"""
        datasets = [
            ("train", self.sents_train, "Answer distribution in the training set"),
            ("test_complex", self.sents_test_complex, "Answer distribution in the test set (complex)"),
            ("test_simple", self.sents_test_simple, "Answer distribution in the test set (simple)")
        ]

        for dataset, sents, title in datasets:
            path = f"results/histograms_answer_counts/{dataset}"
            os.makedirs(path, exist_ok=True)
            plot_histogram(sents, title=title, output_folder=path)

def main():
    # Initialize analyzer
    analyzer = DataAnalyzer()

    #* Download nltk resources (uncomment if needed)
    #analyzer.download_nltk_resources()
    
    # Load and process data
    analyzer.load_datasets()
    
    # Calculate statistics
    analyzer.calculate_sentence_statistics()
    analyzer.calculate_frequency_distributions()
    analyzer.calculate_subject_distributions()
    analyzer.add_vocabulary_sizes()
    
    # Generate plots and visualizations
    analyzer.generate_results()

if __name__ == "__main__":

    main()