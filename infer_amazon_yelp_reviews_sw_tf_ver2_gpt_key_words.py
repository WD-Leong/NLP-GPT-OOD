import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_gpt_keras as tf_gpt
from gpt_utils import extract_key_words

# Model Parameters. #
prob_keep  = 0.9
sub_batch  = 64
num_heads  = 4
num_layers = 3
seq_length = 50

hidden_size = 256
ffwd_size   = 4*hidden_size
cooling_step = 500
weight_decay = 1.0e-4

train_loss_file = "train_loss_amazon_yelp_sw_gpt.csv"
model_data_ckpt_dir = "TF_Models/amazon_yelp_sw_gpt_data"
model_bgrd_ckpt_dir = "TF_Models/amazon_yelp_sw_gpt_bgrd"

# Load the data. #
tmp_pkl_file = "../../Data/amazon-reviews/"
tmp_pkl_file += "amazon_yelp_reviews_subword.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

vocab_size = len(subword_vocab)
print("Vocabulary Size:", str(vocab_size))
del data_tuple

SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Build the GPT. #
print("Building the GPT Model.")
start_time = time.time()

data_model = tf_gpt.GPTDecoder(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    rate1=1.0-prob_keep, rate2=1.0-prob_keep)
data_optimizer = tfa.optimizers.AdamW(
    weight_decay=weight_decay)

bgrd_model = tf_gpt.GPTDecoder(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    rate1=1.0-prob_keep, rate2=1.0-prob_keep)
bgrd_optimizer = tfa.optimizers.AdamW(
    weight_decay=weight_decay)

elapsed_time = (time.time()-start_time) / 60
print("GPT Primer Built", 
      "(" + str(elapsed_time) + " mins).")

# Print the model summary. #
tmp_zero = np.zeros(
    [1, seq_length], dtype=np.int32)
tmp_pred = data_model(tmp_zero, training=True)

print(data_model.summary())
print("-" * 50)
del tmp_zero, tmp_pred

# Create the model checkpoint. #
ckpt_data = tf.train.Checkpoint(
    d_step=tf.Variable(0), 
    data_model=data_model, 
    data_optimizer=data_optimizer)

ckpt_bgrd = tf.train.Checkpoint(
    b_step=tf.Variable(0), 
    bgrd_model=bgrd_model, 
    bgrd_optimizer=bgrd_optimizer)

manager_data = tf.train.CheckpointManager(
    ckpt_data, model_data_ckpt_dir, max_to_keep=1)
manager_bgrd = tf.train.CheckpointManager(
    ckpt_bgrd, model_bgrd_ckpt_dir, max_to_keep=1)

ckpt_data.restore(manager_data.latest_checkpoint)
if manager_data.latest_checkpoint:
    print("Model restored from {}".format(
        manager_data.latest_checkpoint))
else:
    print("Error: No data model checkpoint found.")

ckpt_bgrd.restore(manager_bgrd.latest_checkpoint)
if manager_bgrd.latest_checkpoint:
    print("Model restored from {}".format(
        manager_bgrd.latest_checkpoint))
else:
    print("Error: No bgrd model checkpoint found.")

train_loss_df = pd.read_csv(train_loss_file)
train_loss_list = [tuple(
    train_loss_df.iloc[x].values) \
    for x in range(len(train_loss_df))]

# Infer using the GPT model. #
n_iter = ckpt_data.d_step.numpy().astype(np.int32)
b_iter = ckpt_bgrd.b_step.numpy().astype(np.int32)

print("-" * 50)
print("Inferring the GPT Network")
print("(" + str(n_iter),
      "data iterations,", 
      str(b_iter) + " bgrd iterations.)")
print("-" * 50)

while True:
    tmp_phrase = input("Enter the review: ")
    tmp_phrase = tmp_phrase.lower().strip()

    if tmp_phrase == "":
        break
    else:
        tmp_i_idx = bpe.bp_encode(
            tmp_phrase, subword_vocab, subword_2_idx)
        tmp_i_tok = bpe.bp_decode(tmp_i_idx, idx_2_subword)
        n_sw_toks = len(tmp_i_idx)
        
        tmp_in_phrase = " ".join(
            tmp_i_tok).replace("<", "").replace(">", "")
        
        # Extract the key words. #
        tmp_array = np.array(tmp_i_idx).reshape((1,  -1))
        key_words = extract_key_words(
            data_model, bgrd_model, 
            tmp_array, idx_2_subword, EOS_token)
        del tmp_array
        
        print("")
        print("Input Prompt:")
        print(tmp_in_phrase)
        print("Key Words:")
        print(key_words)
        print("-" * 50)
