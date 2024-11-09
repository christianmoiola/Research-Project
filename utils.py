import json
from collections import Counter
import itertools
import matplotlib.pyplot as plt
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


def plot_word_frequencies(class_dicts, all_classes_freq, top_n=10, mode="total", save_path=None):
    """
    Creates a bar chart with the top n most frequent words based on the chosen mode.
    
    Parameters:
    - class_dicts (dict): Dictionary with keys from 0 to 15, each containing the frequency distribution for a specific class.
    - all_classes_freq (dict): Dictionary containing the total frequency distribution of words.
    - top_n (int): Number of top frequent words to display in the chart.
    - mode (str or int): Specifies whether to select the top words from the total distribution ("total") or a specific class (0-15).
    - save_path (str, optional): File path to save the chart. If None, the chart is displayed instead of saved.
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
    bottom = np.zeros(len(words))  # Base for stacking each column
    
    # Colors for each class
    colors = plt.cm.tab20.colors  # Use a color map with at least 16 different colors
    
    # Add each class to the chart as a stacked segment
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
    ax.set_title(f"Top {top_n} Most Frequent Words in {title_mode}")
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