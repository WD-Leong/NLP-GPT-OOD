# NLP Generative Pre-Training (GPT) Model for Out-of-Distribution (OOD) detection
This repository includes the codes that I have modified for the GPT model, following the publication of [Open AI's GPT Model](https://openai.com/blog/better-language-models/). In particular, the changes include (i) the addition of a learnt positional embedding vector in each layer and (ii) the addition of a residual connection between the input embedding and the output embedding layer.

The Out-of-Distribution (OOD) detection follows that of [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845), where the background model is trained by randomly replacing word tokens (or sub-word tokens) at random. The anomaly score is taken to be the [Kullback-Leibler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the data and background models.

![GPT Model Architecture](GPT_network.png)

Fig. 1: GPT Model Architecture (obtained from [GPT paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf))

This repository includes a code to train the data on the [Movie Dialogue](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset, where the preparation of the data follows this [script](https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/scripts/prepare_data.py) closely. Instead of using a Sequence-to-Sequence model, the dialogue GPT model performs its inference by conditioning on the encoder inputs, followed by the `SOS` token to signal the beginning of the decoder output. For this model, the vocabulary is shared so the token embeddings are the same for both the encoder and decoder.

Before training the model, first process the data by running
```
python process_movie_dialogue_subword.py
```
to use sub-word tokens, followed by
```
python train_dialogue_sw_tf_ver2_gpt.py
```
to train the GPT model. Run the script
```
python infer_dialogue_sw_tf_ver2_gpt.py
```
to perform inference.

For the movie dialogue dataset, the training is done in the following manner - the encoder input is first inserted, followed by the `SOS` token, followed by the decoder output and finally ending with the `EOS` token. For example, if we have
```
Input Phrase:
how are you ?
Output Phrase:
SOS i ' m fine . EOS
```
as the encoder-decoder training pair, the GPT model will transform the encoder and decoder responses into
```
how are you ? SOS i ' m fine . EOS
```
and train the model on the concatenated response. Using the same example, inference is done by setting the seed as the encoder input followed by the `SOS` token, or
```
how are you ? SOS
```
and for illustration purposes, if the GPT model's prediction of the entire sequence is
```
how are you ? SOS i am feeling fine . EOS
```
the decoder response will be taken as the GPT model's prediction following the `SOS` token, which is `i am feeling fine . EOS`.

The GPT model is also trained on the [Reddit Jokes dataset](https://github.com/taivop/joke-dataset). For the Reddit jokes dataset, run the commands
```
python process_reddit_jokes_subword.py
python train_reddit_jokes_sw_tf_ver2_gpt.py
```
to train the model on sub-word tokens. To perform inference, run the command
```
python infer_reddit_jokes_sw_tf_ver2_gpt.py
```
for sub-word tokens model.

To test the effectiveness of extracting key words, the model is also trained on a combined dataset of Amazon and Yelp reviews, made available on [FastAI](https://course.fast.ai/datasets). To train the model, run the commands
```
python process_amazon_yelp_reviews_subword.py
```
to generate the sub-word token vocabulary, and
```
python train_amazon_yelp_reviews_sw_tf_ver2_gpt.py
```
to train the GPT model. After the model is trained, perform inference by running
```
python infer_amazon_yelp_reviews_sw_tf_ver2_gpt.py
```
to generate reviews, or
```
python infer_amazon_yelp_reviews_sw_tf_ver2_gpt_key_words.py
```
to extract the key words of the input text.

## Outputs
As the paper mentioned, the detected anomalies generally refer to semantic components of the text, which is used to identify key words in the input text. Some examples of key words extracted, ranked in order of KL-Divergence scores, are shown below:
```
Input Prompt:
rating_2 its ok fun to shoot but camera angle wont let you see people shooting at you from side can be hard but its fun .
Key Words:
[('shooting', 0.48114390671253204), ('wont', 0.3853304237127304), ('angle', 0.3647970457871755), ('side', 0.3620620295405388), ('see', 0.3555978536605835), ('let', 0.31555381417274475), ('hard', 0.29979565739631653), ('shoot', 0.28081275522708893), ('fun', 0.18501505255699158), ('ok', 0.1440693736076355), ('camera', 0.14065030589699745), ('people', 0.13500372087582946), ('fun', 0.12330926209688187), ('EOS', 0.09144289791584015), ('rating_2', 0.0)]
--------------------------------------------------
Input Prompt:
rating_5 this watch is gorgeous !! very sophisticated and stylish . i ' m so happy that i purchased it - i would recommend this to everyone !!!
Key Words:
[('stylish', 0.44562674686312675), ('gorgeous', 0.23052888568490743), ('everyone', 0.21735593304038048), ('purchased', 0.1934991329908371), ('sophisticated', 0.16780299693346024), ('would', 0.16209185123443604), ('!!', 0.14998765289783478), ('watch', 0.14659184217453003), ('recommend', 0.11900118738412857), ('EOS', 0.10034901648759842), ('happy', 0.08505371864885092), ('!!!', 0.07753344811499119), ('rating_5', 0.0)]
--------------------------------------------------
```
It is also possible to generate text using the GPT models as well:
```
Input Phrase:
what time is it ?
Generated Phrase (Data):
SOS eight o ' clock . EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD
Generated Phrase (Bgrd):
SOS eight o ' clock . EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD
--------------------------------------------------
Input Phrase:
where are we going ?
Generated Phrase (Data):
SOS to the bathroom . EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD
Generated Phrase (Bgrd):
SOS i ' m going to miss you . EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD
--------------------------------------------------
Input Phrase:
how much is it ?
Generated Phrase (Data):
SOS four hundred thousand . EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD
Generated Phrase (Bgrd):
SOS two hundred pounds . EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD
--------------------------------------------------
```
To generate text, the data model is usually used although the background model can sometimes give interesting results.

## Miscellaneous
The models were trained using Intel i5 CPU. It is possible to improve the training times with the Intel MKL optimised Tensorflow version 2 libraries. It can be installed via
```
conda install tensorflow -c intel
```
using Anaconda in Windows 10.
