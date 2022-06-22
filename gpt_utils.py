import numpy as np
import tensorflow as tf
from string import punctuation
from nltk.corpus import stopwords

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, 
    x_encode, x_output, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_encode = x_encode[id_st:id_en, :]
        tmp_output = x_output[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            output_logits = model(tmp_encode)
            
            tmp_losses = tf.reduce_sum(tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=output_logits), axis=1))
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [
            (acc_grad + grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Divide the accumulated gradients by the batch size. #
    batch_losses  = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    # Update using the optimizer. #
    clip_tuple = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clip_tuple[0], model_params))
    return batch_losses

# Compute the KL-Divergence. #
def compute_kl_div(
    data_model, bgrd_model, x_input, eps=1.0e-9):
    batch_size = x_input.shape[0]
    seq_length = x_input.shape[1]
    
    kl_div_scores = np.zeros(
        [batch_size, seq_length+1], dtype=np.float32)
    for n_seq in range(seq_length):
        data_probs = tf.nn.softmax(data_model(
            x_input[:, :(n_seq+1)], training=False))
        bgrd_probs = tf.nn.softmax(bgrd_model(
            x_input[:, :(n_seq+1)], training=False))
        bgrd_probs = bgrd_probs + eps

        # Compute the score. #
        raw_score = np.multiply(
            data_probs[:, -1, :], np.log(np.divide(
                data_probs[:, -1, :], bgrd_probs[:, -1, :])))
        kl_div_scores[:, n_seq+1] = np.sum(raw_score, axis=1)
    return kl_div_scores

# Average the anomaly score across sub-word tokens. #
def bp_kl_decode(tuple_in, idx2subword):
    """
    tuple_in should take on the form (subword_index, kl_score).
    """
    sw_kl_score = [y for x, y in tuple_in]
    sw_idx_list = [
        idx2subword[x] for x, y in tuple_in]
    words_list  = []
    
    n_curr  = 0.0
    curr_sw = ""
    curr_kl = 0.0
    for n_sw in range(len(sw_idx_list)):
        tmp_sw = sw_idx_list[n_sw]
        tmp_kl = sw_kl_score[n_sw]

        if tmp_sw.find("<") != -1 \
            and tmp_sw.find(">") != -1:
            # Sub-word is a word. #
            # Average score is the score. #
            tmp_word  = tmp_sw
            tmp_score = tmp_kl
            avg_score = tmp_score

            n_curr  = 0
            curr_sw = ""
            curr_kl = 0.0
            words_list.append((tmp_word, avg_score))
        elif tmp_sw.find(">") != -1 \
            and tmp_sw.find("<") == -1:
            # End of Sub-word. #
            n_curr  += 1
            curr_sw += tmp_sw
            curr_kl += tmp_kl

            tmp_word  = curr_sw
            tmp_score = curr_kl
            avg_score = tmp_score / n_curr

            n_curr  = 0
            curr_sw = ""
            curr_kl = 0.0
            words_list.append((tmp_word, avg_score))
        elif tmp_sw.find(">") == -1 \
            and tmp_sw.find("<") != -1:
            # Start of Sub-word. #
            n_curr  += 1
            curr_sw += tmp_sw
            curr_kl += tmp_kl
        else:
            # Continuation of Sub-word. #
            n_curr  += 1
            curr_sw += tmp_sw
            curr_kl += tmp_kl
    return words_list

def extract_key_words(
    data_model, bgrd_model, x_input, 
    idx_2_subword, EOS_token, eps=1.0e-9):
    en_stopwords = stopwords.words('english')
    exclude_list = en_stopwords
    exclude_list += [x for x in punctuation]

    # Get the KL-Divergence score. #
    batch_size = x_input.shape[0]
    seq_length = x_input.shape[1]
    if batch_size > 1:
        print("Error: Current implementation only supports one test case.")
        return None
    
    kl_div_scores = np.zeros(
        [batch_size, seq_length+1], dtype=np.float32)
    for n_seq in range(seq_length):
        data_probs = tf.nn.softmax(data_model(
            x_input[:, :(n_seq+1)], training=False))
        bgrd_probs = tf.nn.softmax(bgrd_model(
            x_input[:, :(n_seq+1)], training=False))
        
        # Prevent errors in division or log. #
        data_probs = data_probs + eps
        bgrd_probs = bgrd_probs + eps

        # Compute the score. #
        raw_score = np.multiply(
            data_probs[:, -1, :], np.log(np.divide(
                data_probs[:, -1, :], bgrd_probs[:, -1, :])))
        kl_div_scores[:, n_seq+1] = np.sum(raw_score, axis=1)
    
    # Generate the array of (subword_index, score). #
    tmp_scores  = kl_div_scores[0]
    tmp_i_index = list(x_input[0]) + [EOS_token]
    kl_tuple = [(
        tmp_i_index[x], tmp_scores[x]) \
            for x in range(seq_length+1)]
    
    kl_decode = bp_kl_decode(kl_tuple, idx_2_subword)
    kl_scores = np.array([x[1] for x in kl_decode])
    kl_words  = [str(x[0]).replace(
        "<", "").replace(">", "") for x in kl_decode]
    
    # Remove stop words. #
    out_length = len(kl_words)
    idx_sorted = np.argsort(-kl_scores)
    words_sort = [(
        kl_words[idx_sorted[x]], 
        kl_scores[idx_sorted[x]]) \
            for x in range(out_length) if \
                kl_words[idx_sorted[x]] not in exclude_list]
    return words_sort
