# Import the libraries. #
import time
import collections
import pandas as pd
import pickle as pkl
from collections import Counter
import byte_pair_encoding as bpe
from nltk.tokenize import wordpunct_tokenize as word_tokenizer

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split(" ")
        for m in range(len(symbols)-1):
            pairs[symbols[m], symbols[m+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = " ".join(pair)
    for word in v_in:
        w_out = word.replace(bigram, "".join(pair))
        v_out[w_out] = v_in[word]
    return v_out

def learn_subword_vocab(word_vocab, n_iter):
    for m in range(n_iter):
        pairs = get_stats(word_vocab)
        most_freq  = max(pairs, key=pairs.get)
        word_vocab = merge_vocab(most_freq, word_vocab)
    
    subword_vocab = collections.defaultdict(int)
    for word, freq in pairs.items():
        if freq == 0:
            continue
        subword_vocab[word[0]] += freq
        subword_vocab[word[1]] += freq
    return subword_vocab, pairs

print("Loading the amazon reviews data.")
start_tm = time.time()

tmp_csv_file = "../../../Data/amazon-reviews/"
tmp_csv_file += "amazon_review_full_csv/train.csv"
tmp_csv_data = pd.read_csv(tmp_csv_file, header=0)
print("Total of", len(tmp_csv_data), "reviews loaded.")

# Populate the reviews. #
tmp_csv_data.columns = ["rating", "summary", "review"]
print(tmp_csv_data.head())

print("Loading the yelp reviews data.")
start_tm = time.time()

tmp_csv_file = "../../../Data/yelp-reviews/"
tmp_csv_file += "yelp_review_full_csv/train.csv"
tmp_new_data = pd.read_csv(tmp_csv_file, header=0)
print("Total of", len(tmp_csv_data), "reviews loaded.")

# Populate the reviews. #
tmp_new_data.columns = ["rating", "review"]
print(tmp_new_data.head())

# Append the two datasets. #
tmp_csv_data.drop(
    ["summary"], axis=1, inplace=True)
tmp_csv_data = tmp_csv_data.append(
    tmp_new_data, ignore_index=True)

print("Total of", len(tmp_csv_data), "combined reviews loaded.")
del tmp_new_data

# Pre-processing. #
max_len = 30

tmp_data  = []
num_data  = len(tmp_csv_data)
w_counter = Counter()
for n in range(len(tmp_csv_data)):
    tmp_row = tmp_csv_data.iloc[n]

    tmp_rating = "rating_" + str(tmp_row["rating"])
    tmp_review = tmp_row["review"]
    tmp_review = tmp_review.lower()
    
    # Minor data cleaning for Yelp reviews. #
    tmp_review = tmp_review.replace("\n", " ")
    tmp_review = tmp_review.replace("\ n", " ")

    tmp_tokens = [tmp_rating]
    tmp_tokens += [
        x for x in word_tokenizer(tmp_review) if x != ""]
    
    if len(tmp_tokens) == 0:
            continue
    elif len(tmp_tokens) <= max_len:
        w_counter.update(tmp_tokens)
        tmp_data.append(" ".join(tmp_tokens))
    
    if (n+1) % int(num_data/10) == 0:
        print(n+1, "out of", num_data, "reviews pre-processed.")
del tmp_csv_data

elapsed_tm = (time.time() - start_tm) / 60
print("Total of", len(tmp_data), "reviews filtered.")
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Fit the subword vocabulary. #
print("Fitting subword vocabulary.")
start_tm = time.time()

word_counts = []
for word, count in w_counter.items():
    tmp_word = "<" + word + ">"
    tmp_word = "".join(
        [x + " " for x in tmp_word]).strip()
    word_counts.append((tmp_word, count))
word_counts = dict(word_counts)

n_iters = 500
vocab_size = 12000
tuple_out  = bpe.learn_subword_vocab(
    word_counts, n_iters, vocab_size=vocab_size)

subword_vocab = tuple_out[0]
idx_2_subword = tuple_out[1]
subword_2_idx = tuple_out[2]

elapsed_tm = (time.time() - start_tm) / 60
print("Total Sub-word Vocabulary size:", 
      len(subword_vocab), "sub-word tokens.")
print("Elapsed Time:", round(elapsed_tm, 2), "mins.")

# Encode the corpus to subword tokens. #
print("Encoding the corpus to subwords.")
start_tm = time.time()

sw_tuple = []
num_data = len(tmp_data)
for n in range(num_data):
    tmp_review = tmp_data[n]

    tmp_sw = bpe.bp_encode(
        tmp_review, subword_vocab, subword_2_idx)
    sw_tuple.append(tmp_sw)

    if (n+1) % int(num_data / 10) == 0:
        print(n+1, "out of", num_data, "reviews encoded.")

elapsed_tm = (time.time() - start_tm) / 60
print("Elapsed Time:", round(elapsed_tm, 2), "mins.")

# Save the data. #
print("Saving the file.")
tmp_pkl_file = "../../../Data/amazon-reviews/"
tmp_pkl_file += "amazon_yelp_reviews_subword.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(sw_tuple, tmp_file_save)
    pkl.dump(subword_vocab, tmp_file_save)
    pkl.dump(idx_2_subword, tmp_file_save)
    pkl.dump(subword_2_idx, tmp_file_save)
