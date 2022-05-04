from urllib.request import urlretrieve
import os

if __name__ == "__main__":
    if not os.path.isfile('datasets/mini.h5'):
        print("Downloading Conceptnet Numberbatch word embeddings...")
        conceptnet_url = 'http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/17.06/mini.h5'
        urlretrieve(conceptnet_url, 'datasets/mini.h5')

    # Load the file and pull out words and embeddings
    import h5py

    with h5py.File('datasets/mini.h5', 'r') as f:
        all_words = [word.decode('utf-8') for word in f['mat']['axis1'][:]]
        all_embeddings = f['mat']['block0_values'][:]

    # Restrict our vocabulary to just the English words
    english_words = [word[6:] for word in all_words if word.startswith('/c/en/')]
    english_word_indices = [i for i, word in enumerate(all_words) if word.startswith('/c/en/')]
    english_embeddings = all_embeddings[english_word_indices]

    import numpy as np

    norms = np.linalg.norm(english_embeddings, axis=1)
    normalized_embeddings = english_embeddings.astype('float32') / norms.astype('float32').reshape([-1, 1])
    index = {word: i for i, word in enumerate(english_words)}


    def closest_to_vector(v, n):
        all_scores = np.dot(normalized_embeddings, v)
        best_words = list(map(lambda i: english_words[i], reversed(np.argsort(all_scores))))
        return best_words[:n]


    def most_similar(w, n):
        return closest_to_vector(normalized_embeddings[index[w], :], n)


    import string

    remove_punct = str.maketrans('', '', string.punctuation)


    # This function converts a line of our data file into
    # a tuple (x, y), where x is 300-dimensional representation
    # of the words in a review, and y is its label.
    def convert_line_to_example(line):
        # Split the line into words using Python's split() function
        words = line.translate(remove_punct).lower().split()

        # Look up the embeddings of each word, ignoring words not
        # in our pretrained vocabulary.
        embeddings = [normalized_embeddings[index[w]] for w in words
                    if w in index]

        # Take the mean of the embeddings
        x = np.mean(np.vstack(embeddings), axis=0)
        return x


    def similarity_score(w1, w2):
        score = np.dot(convert_line_to_example(w1), convert_line_to_example(w2))
        return score
