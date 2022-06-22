# Import the libraries. #
import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_gpt_keras as gpt
import byte_pair_encoding as bpe
from gpt_utils import (
    compute_kl_div, bp_kl_decode)

# Model Parameters. #
seq_length = 31
num_heads  = 4
num_layers = 3
prob_keep  = 0.9

hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 200

model_data_ckpt_dir = "TF_Models/gpt_data_sw_dialogue"
model_bgrd_ckpt_dir = "TF_Models/gpt_bgrd_sw_dialogue"

# Load the data. #
tmp_pkl_file = "../../Data/movie_dialogs/"
tmp_pkl_file += "movie_dialogues_subword.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

vocab_size = len(subword_2_idx)
print("Vocabulary Size:", str(vocab_size) + ".")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

num_data  = len(data_tuple)
SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]
print("Total of", str(num_data), "rows loaded.")

# Build the GPT model. #
print("Building the GPT Model.")
start_time = time.time()

# For the data model. #
data_model = gpt.GPTDecoder(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    rate1=1.0-prob_keep, rate2=1.0-prob_keep)
data_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

# For the background model. #
bgrd_model = gpt.GPTDecoder(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    rate1=1.0-prob_keep, rate2=1.0-prob_keep)
bgrd_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

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
del tmp_outputs
print(data_model.summary())

n_iter = ckpt_data.d_step.numpy().astype(np.int32)
b_iter = ckpt_bgrd.b_step.numpy().astype(np.int32)

print("-" * 50)
print("Training the GPT Models.")
print("(" + str(n_iter),
      "data iterations,", 
      str(b_iter) + " bgrd iterations.)")
print(str(num_data), "training samples.")
print("-" * 50)

while True:
    tmp_prompt = input("Enter prompt: ")
    tmp_prompt = tmp_prompt.lower().strip()

    if tmp_prompt == "":
        break
    else:
        tmp_test_in[:, :] = PAD_token

        tmp_i_index  = bpe.bp_encode(
            tmp_prompt, subword_vocab, subword_2_idx)
        tmp_p_index  = tmp_i_index + [SOS_token]
        n_seq_length = len(tmp_p_index)
        
        tmp_i_tok = bpe.bp_decode(
            tmp_i_index, idx_2_subword)
        n_tokens  = len(tmp_i_index) + 1
        
        tmp_in_phrase  = " ".join(
            tmp_i_tok).replace("<", "").replace(">", "")

        if n_seq_length > seq_length:
            n_seq_toks = seq_length
            tmp_test_in[0, :] = tmp_p_index[:seq_length]
        else:
            n_seq_toks = n_seq_length
            tmp_test_in[0, :n_seq_length] = tmp_p_index
        
        # Infer the generated sequence. #
        tmp_data_infer = data_model.infer(
            tmp_test_in[:, :n_seq_toks]).numpy()
        tmp_bgrd_infer = bgrd_model.infer(
            tmp_test_in[:, :n_seq_toks]).numpy()

        # Compute the KL-Divergence. #
        tmp_kl_div = compute_kl_div(
            data_model, bgrd_model, 
            tmp_data_infer[:, :-1])[0]
        
        tmp_seq  = list(tmp_data_infer[0])
        kl_tuple = [(
            tmp_seq[x], tmp_kl_div[x]) \
                for x in range(seq_length+1)]
        kl_display = bp_kl_decode(kl_tuple, idx_2_subword)
        
        # Decode the subwords. #
        gen_phrase = bpe.bp_decode(
            tmp_data_infer[0, (n_tokens-1):], idx_2_subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        
        bgrd_phrase = bpe.bp_decode(
            tmp_bgrd_infer[0, (n_tokens-1):], idx_2_subword)
        bgrd_phrase = " ".join(
            bgrd_phrase).replace("<", "").replace(">", "")
        del tmp_p_index, n_tokens, tmp_i_index
        
        print("")
        print("Input Phrase:")
        print(tmp_in_phrase)
        print("Generated Phrase (Data):")
        print(gen_phrase)
        print("Generated Phrase (Bgrd):")
        print(bgrd_phrase)
        print("KL-Divergence Scores:")
        print(kl_display)
        print("-" * 50)
