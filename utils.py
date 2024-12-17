import json
from collections import Counter
import itertools
import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np
import os
# Function to load the data from a json file
def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def flatten_list(data):
    '''
        input: list
        output: list
    '''
    return list(itertools.chain.from_iterable(data))

def group_sents(sents):
    if "all_classes" in sents:
        return sents
    ret = {"all_classes": []}
    for key in sents.keys():
        ret["all_classes"].extend(sents[key])
    return ret


def create_table(data, colums, path):
    fig, ax = plt.subplots(figsize=(12, 8)) 
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])

    n_rows, n_cols = len(data)+1, len(colums)
    width, height = 1.0 / n_cols, 1.0 / n_rows

    # Add the column names
    for i, col in enumerate(colums):
        table.add_cell(0, i, width, height, text=col, loc='center', facecolor='lightgray', edgecolor='black')

    # Add the data
    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            table.add_cell(i+1, j, width, height, text=cell, loc='center', facecolor='white', edgecolor='black')

    for i in range(n_rows):
        for j in range(n_cols):
            #table[(i, j)].set_fontsize(20)
            table[(i, j)].set_edgecolor('black')

    ax.add_table(table)
    plt.savefig(path, bbox_inches='tight', dpi=300)    


def plot_word_frequencies(class_dicts, all_classes_freq, top_n=10, mode="total", save_path=None, title=None, single_class_mode=True):
    """
    Creates a bar chart with the top n most frequent words based on the chosen mode.

    Parameters:
    - class_dicts (dict): Dictionary with keys from 0 to 15, each containing the frequency distribution for a specific class.
    - all_classes_freq (dict): Dictionary containing the total frequency distribution of words.
    - top_n (int): Number of top frequent words to display in the chart.
    - mode (str or int): Specifies whether to select the top words from the total distribution ("total") or a specific class (0-15).
    - save_path (str, optional): File path to save the chart. If None, the chart is displayed instead of saved.
    - single_class_mode (bool): If True, only the frequencies for the specified class (via mode) or total distribution are displayed.
    """
    # Select the reference distribution based on the mode
    if mode == "total":
        reference_freq = all_classes_freq
    elif isinstance(mode, int) and 0 <= mode <= 15:
        reference_freq = class_dicts.get(mode, {})
    else:
        raise ValueError("The 'mode' parameter should be 'total' or an integer between 0 and 15.")

    # Get the top_n most frequent words from the selected distribution
    most_common_words = sorted(reference_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words = [word for word, _ in most_common_words]

    # Prepare data for each class and for each selected word
    class_frequencies = {word: [class_dicts[class_id].get(word, 0) for class_id in range(16)] for word in words}

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    if single_class_mode:
        # Single class mode: Plot only the specified class or total distribution
        if mode == "total":
            single_class_frequencies = [reference_freq.get(word, 0) for word in words]
            ax.bar(words, single_class_frequencies, color="skyblue", label="Total Distribution")
        elif isinstance(mode, int) and 0 <= mode <= 15:
            single_class_frequencies = [class_frequencies[word][mode] for word in words]
            ax.bar(words, single_class_frequencies, color="skyblue", label=f'Class {mode}')
        else:
            raise ValueError("When 'single_class_mode' is True, 'mode' must be 'total' or an integer between 0 and 15.")
    else:
        # Multi-class mode: Stack frequencies by class
        bottom = np.zeros(len(words))  # Base for stacking each column
        colors = plt.cm.tab20.colors  # Use a color map with at least 16 different colors

        for class_id in range(16):
            heights = [class_frequencies[word][class_id] for word in words]
            if len(heights) != len(bottom):
                heights = heights[:len(bottom)]  # Trim if necessary to match `bottom`
            ax.bar(words, heights, bottom=bottom, color=colors[class_id % len(colors)], label=f'Class {class_id}')
            bottom += heights  # Update the base for the next segment

    # Configure the chart
    title_mode = "All Classes" if mode == "total" else f"Class {mode}"
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    if title is None:
        ax.set_title(f"Top {top_n} Most Frequent Words in {title_mode}")
    else:
        ax.set_title(title)

    if not single_class_mode:
        ax.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot if save_path is provided, otherwise display it
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    else:
        plt.show()


def plot_histogram(dictionary, output_folder, title="Histogram"):
    """
    Creates and saves a histogram showing the number of sentences for each class.

    Args:
        dictionary (dict): Dictionary where keys are classes and values are lists of sentences.
        output_folder (str): Path to the folder where the histogram image will be saved.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract classes and sentence counts
    classes = list(dictionary.keys())
    counts = [len(sentences) for sentences in dictionary.values()]

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Number of Sentences')
    plt.title(title)
    plt.xticks(rotation=45)
    
    # Save the histogram in the specified folder
    output_path = os.path.join(output_folder, 'class_histogram.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram saved in: {output_path}")

def vocabulary_lengths(dictionary):
    '''
    Length of the vocabulary for each class in the dictionary.
    '''
    result = []
    for class_name, vocab_dict in dictionary.items():
        vocab_size = len(vocab_dict)
        result.append(f"Class '{class_name}' has a vocabulary size of {vocab_size}")
    
    return "\n".join(result)

def filter_freq_dist(dictionary, pos_tag):
    '''
    Function to filter words by a specific POS tag in each class, assuming the dictionary is already POS-tagged.

    Parameters:
    - dictionary (dict): Dictionary with POS tags as keys and words as values for each class.
    - pos_tag (str): The POS tag to filter by (e.g., 'NN', 'JJ', etc.).

    Output:
    - filtered_dict (dict): Dictionary with filtered words based on the specified POS tag for each class.
    '''
    filtered_dict = {}
    
    for class_name, pos_dict in dictionary.items():
        # Filter words that match the specified POS tag
        filtered_words = pos_dict.get(pos_tag, {})
        filtered_dict[class_name] = filtered_words
    
    return filtered_dict

def flatten_freq_distribution(freq_dist):
    '''
    Flatten the frequency distribution of different classes into a single dictionary with the key 'all_classes'.
    '''
    combined = {"all_classes": Counter()}
    for key, class_freqs in freq_dist.items():
        combined["all_classes"].update(class_freqs)
    combined["all_classes"] = dict(combined["all_classes"])  # Converti in dizionario normale
    return combined