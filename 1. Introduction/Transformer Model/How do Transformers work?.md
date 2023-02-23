# [How do Transformers work? - Hugging Face Course](https://huggingface.co/course/chapter1/4?fw=pt)

## Red
* general pretrained model then goes through a process called transfer learning. During this process, the model is fine-tuned in a supervised way — that is, using human-annotated labels — on a given task.

* All the Transformer models mentioned above (GPT, BERT, BART, T5, etc.) have been trained as language models.

## Green
* This is why sharing language models is paramount: sharing the trained weights and building on top of already trained weights reduces the overall compute cost and carbon footprint of the community

## Red
* Pretraining is the act of training a model from scratch

* Fine-tuning, on the other hand, is the training done after a model has been pretrained

* o perform fine-tuning, you first acquire a pretrained language model, then perform additional training with a dataset specific to your task

## Yellow
* The pretrained model was already trained on a dataset that has some similarities with the fine-tuning dataset. The fine-tuning process is thus able to take advantage of knowledge acquired by the initial model during pretraining (for instance, with NLP problems, the pretrained model will have some kind of statistical understanding of the language you are using for your task). Since the pretrained model was already trained on lots of data, the fine-tuning requires way less data to get decent results. For the same reason, the amount of time and resources needed to get good results are much lower.

## Orange
* the knowledge the pretrained model has acquired is “transferred,” hence the term transfer learning.

## Red
* Fine-tuning a model therefore has lower time, data, financial, and environmental costs

* over different fine-tuning schemes

* General architecture

## Yellow
* Encoder (left): The encoder receives an input and builds a representation of it (its features).

* Decoder (right): The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence

## Purple
* Each of these parts can be used independently, depending on the task:

* Encoder-only models: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition. Decoder-only models: Good for generative tasks such as text generation. Encoder-decoder models or sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.

## Grey
* Attention layers

* attention layers

* this layer will tell the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others) when dealing with the representation of each word.

* English to French

* The same concept applies to any task associated with natural language: a word by itself has a meaning, but that meaning is deeply affected by the context, which can be any other word (or words) before or after the word being studied.

## Red
* The original architecture

* The Transformer architecture was originally designed for translation.

* In the encoder, the attention layers can use all the words in a sentence (since, as we just saw, the translation of a given word can be dependent on what is after as well as before it in the sentence)

* The decoder

* can only pay attention to the words in the sentence that it has already translated

* The attention mask can also be used in the encoder/decoder to prevent the model from paying attention to some special words — for instance, the special padding word used to make all the inputs the same length when batching together sentences.

## Orange
* Architectures vs. checkpoints

* Architecture: This is the skeleton of the model

* Checkpoints: These are the weights that will be loaded in a given architecture.

* Model: This is an umbrella term that isn’t as precise as “architecture” or “checkpoint”: it can mean both.

* For example, BERT is an architecture while bert-base-cased, a set of weights trained by the Google team for the first release of BERT, is a checkpoint. However, one can say “the BERT model” and “the bert-base-cased model.”

---
Created with [Super Simple Highlighter](https://chrome.google.com/webstore/detail/hhlhjgianpocpoppaiihmlpgcoehlhio). (C)2010-19 [Dexterous Logic](https://www.dexterouslogic.com)