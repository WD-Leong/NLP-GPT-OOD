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
    sub_batch_train_step, compute_kl_div, 
    bp_kl_decode, extract_key_words)

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_length = 31

num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 10000
restore_flag  = True
save_step     = 100
warmup_steps  = 3000
display_step  = 50
anneal_step   = 2500
anneal_rate   = 0.75

prob_keep = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 200

train_loss_file = "train_loss_gpt_sw_dialogue.csv"
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
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Filter the dataset. #
filtered_data = []
for tmp_data in data_tuple:
    tot_len = len(tmp_data[0]) + len(tmp_data[1])
    if tot_len > 1 and tot_len <= (seq_length-1):
        filtered_data.append(tmp_data)
del tmp_data, data_tuple

data_tuple = filtered_data
del filtered_data

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
print("GPT Models Built", 
      "(" + str(elapsed_time), "mins).")

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

if restore_flag:
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
else:
    print("Training a new model.")
    train_loss_list = []

# Train the GPT model. #
tmp_test_in  = np.zeros(
    [1, seq_length], dtype=np.int32)
tmp_data_seq = np.zeros(
    [batch_size, seq_length+1], dtype=np.int32)
tmp_bgrd_seq = np.zeros(
    [batch_size, seq_length+1], dtype=np.int32)

# Print the GPT model summary. #
tmp_outputs = data_model(
    tmp_test_in, training=True)
del tmp_outputs
print(data_model.summary())

# Warmup learning schedule. #
n_iter = ckpt_data.d_step.numpy().astype(np.int32)
b_iter = ckpt_bgrd.b_step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.001
    tmp_learnrate = np.multiply(
        anneal_rate**(n_iter // anneal_step), initial_lr)
    learning_rate = max(tmp_learnrate, 1.0e-5)

print("-" * 50)
print("Training the GPT Models.")
print("(" + str(n_iter),
      "data iterations,", 
      str(b_iter) + " bgrd iterations.)")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_data_loss = 0.0
tot_bgrd_loss = 0.0

start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        if n_iter % anneal_step == 0:
            tmp_learnrate = np.multiply(
                anneal_rate**(n_iter // anneal_step), initial_lr)
            learning_rate = max(tmp_learnrate, 1.0e-6)
    
    # Select a random sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_data_seq[:, :] = PAD_token
    tmp_bgrd_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_i_idx = data_tuple[tmp_index][0]
        tmp_o_idx = data_tuple[tmp_index][1]
        
        tmp_p_idx = tmp_i_idx + [SOS_token]
        tmp_p_idx += tmp_o_idx + [EOS_token]
        n_sw_toks = len(tmp_p_idx)
        
        tmp_mask  = np.random.binomial(
            1, 0.10, size=n_sw_toks)
        tmp_noise = np.random.choice(
            vocab_size, size=n_sw_toks)
        tmp_b_idx = [
            tmp_p_idx[x] if tmp_mask[x] == 0 else \
                tmp_noise[x] for x in range(n_sw_toks)]
        
        tmp_data_seq[n_index, :n_sw_toks] = tmp_p_idx
        tmp_bgrd_seq[n_index, :n_sw_toks] = tmp_b_idx
        del tmp_i_idx, tmp_o_idx, tmp_index
        del tmp_p_idx, tmp_b_idx, tmp_mask, tmp_noise
    
    # Set the training data. #
    tmp_data_input = tmp_data_seq[:, :-1]
    tmp_bgrd_input = tmp_bgrd_seq[:, :-1]
    tmp_data_output = tmp_data_seq[:, 1:]
    tmp_bgrd_output = tmp_bgrd_seq[:, 1:]
    
    tmp_data_loss = sub_batch_train_step(
        data_model, sub_batch, 
        tmp_data_input, tmp_data_output, data_optimizer, 
        learning_rate=learning_rate, grad_clip=gradient_clip)
    
    tmp_bgrd_loss = sub_batch_train_step(
        bgrd_model, sub_batch, 
        tmp_bgrd_input, tmp_bgrd_output, bgrd_optimizer, 
        learning_rate=learning_rate, grad_clip=gradient_clip)
    
    # Update the step. #
    n_iter += 1
    ckpt_data.d_step.assign_add(1)
    ckpt_bgrd.b_step.assign_add(1)
    
    # Display the training progress. #
    tot_data_loss += tmp_data_loss.numpy()
    tot_bgrd_loss += tmp_bgrd_loss.numpy()
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_data_loss = tot_data_loss / display_step
        avg_bgrd_loss = tot_bgrd_loss / display_step
        tot_data_loss = 0.0
        tot_bgrd_loss = 0.0
        
        tmp_test_in[:, :] = PAD_token
        sample_test = np.random.choice(num_data)
        tmp_i_index = data_tuple[sample_test][0]
        tmp_o_index = data_tuple[sample_test][1]

        tmp_p_index = tmp_i_index + [SOS_token]
        tmp_p_index += tmp_o_index
        n_seq_length = len(tmp_p_index)
        
        tmp_i_tok = bpe.bp_decode(
            tmp_i_index, idx_2_subword)
        tmp_o_tok = bpe.bp_decode(
            tmp_o_index, idx_2_subword)
        n_tokens  = len(tmp_i_index) + 1
        
        tmp_in_phrase  = " ".join(
            tmp_i_tok).replace("<", "").replace(">", "")
        tmp_out_phrase = " ".join(
            tmp_o_tok).replace("<", "").replace(">", "")

        if n_seq_length > seq_length:
            n_seq_toks = seq_length
            tmp_test_in[0, :] = tmp_p_index[:seq_length]
        else:
            n_seq_toks = n_seq_length
            tmp_test_in[0, :n_seq_length] = tmp_p_index
        
        # Infer the generated sequence. #
        tmp_data_infer = data_model.infer(
            tmp_test_in[:, :n_tokens]).numpy()[0]
        tmp_bgrd_infer = bgrd_model.infer(
            tmp_test_in[:, :n_tokens]).numpy()[0]
        del sample_test

        # Compute the KL-Divergence. #
        tmp_kl_div = compute_kl_div(
            data_model, bgrd_model, tmp_test_in)[0]
        
        kl_tuple = [(
            tmp_p_index[x], tmp_kl_div[x]) \
                for x in range(n_seq_length)]
        kl_display = bp_kl_decode(kl_tuple, idx_2_subword)

        # Extract the key words. #
        tmp_array = np.array(tmp_p_index).reshape(1, -1)
        key_words = extract_key_words(
            data_model, bgrd_model, 
            tmp_array, idx_2_subword, EOS_token)
        del tmp_array
        
        # Decode the subwords. #
        gen_phrase = bpe.bp_decode(
            tmp_data_infer[(n_tokens-1):], idx_2_subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        
        bgrd_phrase = bpe.bp_decode(
            tmp_bgrd_infer[(n_tokens-1):], idx_2_subword)
        bgrd_phrase = " ".join(
            bgrd_phrase).replace("<", "").replace(">", "")
        del tmp_p_index, n_tokens, tmp_i_index, tmp_o_index
        
        elapsed_tm = (end_tm - start_tm) / 60.0
        print("Iteration", str(n_iter)+".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip) + ".")
        print("Learning Rate:", str(learning_rate) + ".")

        print("Average Data Loss:", str(avg_data_loss) + ".")
        print("Average Bgrd Loss:", str(avg_bgrd_loss) + ".")
        
        print("")
        print("Input Phrase:")
        print(tmp_in_phrase)
        print("Generated Phrase (Data):")
        print(gen_phrase)
        print("Generated Phrase (Bgrd):")
        print(bgrd_phrase)
        print("Actual Response:")
        print(tmp_out_phrase)
        print("KL-Divergence Scores:")
        print(kl_display)
        print("Key words:")
        print(key_words)
        
        train_loss_list.append((
            n_iter, avg_data_loss, avg_bgrd_loss))
        start_tm = time.time()
        print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_data_path = manager_data.save()
        save_bgrd_path = manager_bgrd.save()
        print("Saved data model to {}".format(save_data_path))
        print("Saved background model to {}".format(save_bgrd_path))
        
        tmp_df_cols = ["n_iter", "xent_data", "xent_bgrd"]
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=tmp_df_cols)
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)

