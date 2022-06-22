# Import the libraries. #
import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tf_ver2_gpt_keras as gpt
import byte_pair_encoding as bpe
from gpt_utils import extract_key_words

# Model Parameters. #
seq_length = 51
num_heads  = 4
num_layers = 3
prob_keep  = 0.9

hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 100

model_data_ckpt_dir = "TF_Models/gpt_data_subword_reddit"
model_bgrd_ckpt_dir = "TF_Models/gpt_bgrd_subword_reddit"

# Load the data. #
tmp_pkl_file = "../../Data/reddit_jokes/"
tmp_pkl_file += "reddit_jokes_subword_v1.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

vocab_size = len(subword_2_idx)
print("Vocabulary Size:", str(vocab_size) + ".")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]
print("Total of", str(len(full_data)), "rows loaded.")

# Build the GPT model. #
print("Building the GPT Model.")
start_time = time.time()

# For the data model. #
data_model = gpt.GPTDecoder(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    rate1=1.0-prob_keep, rate2=1.0-prob_keep)
data_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

# For the background model. #
bgrd_model = gpt.GPTDecoder(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    rate1=1.0-prob_keep, rate2=1.0-prob_keep)
bgrd_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time()-start_time) / 60
print("GPT Models Built", "(" + str(elapsed_time), "mins).")

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

# GPT model inference. #
tmp_test_in  = np.zeros(
    [1, seq_length], dtype=np.int32)

# Print the GPT model summary. #
tmp_outputs = data_model(
    tmp_test_in, training=True)

print(data_model.summary())
del tmp_outputs

# Warmup learning schedule. #
n_iter = ckpt_data.d_step.numpy().astype(np.int32)
b_iter = ckpt_bgrd.b_step.numpy().astype(np.int32)

print("-" * 50)
print("GPT Model Inference", 
      "(" + str(n_iter), 
      "data iterations,", 
      str(b_iter) + " bgrd iterations).")
print("-" * 50)

while True:
    tmp_prompt = input("Enter prompt: ")
    tmp_prompt = tmp_prompt.lower().strip()
    if tmp_prompt == "":
        break
    else:
        tmp_i_index = bpe.bp_encode(
            tmp_prompt, subword_vocab, subword_2_idx)
        
        in_phrase = bpe.bp_decode(
            tmp_i_index, idx_2_subword)
        in_phrase = " ".join(
            in_phrase).replace("<", "").replace(">", "")
        
        # Convert the input text into an array. #
        tmp_test_in = np.array(tmp_i_index)
        tmp_test_in = tmp_test_in.reshape(1, -1)
        
        key_words = extract_key_words(
            data_model, bgrd_model, 
            tmp_test_in, idx_2_subword, EOS_token)
        del tmp_i_index
        
        print("")
        print("Input Phrase:")
        print(in_phrase)
        print("Key words:")
        print(key_words)
        print("-" * 50)
