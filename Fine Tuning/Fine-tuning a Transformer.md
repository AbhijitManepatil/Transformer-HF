# [Fine-tuning a model with the Trainer API - Hugging Face](https://huggingface.co/course/chapter3/3?fw=pt)

## Fine-tuning a model with the Trainer API
* Fine-tuning a model with the Trainer API

* Transformers provides a Trainer class to help you fine-tune any of the pretrained models it provides on your dataset.

* The first step before we can define our Trainer is to define a TrainingArguments class that will contain all the hyperparameters the Trainer will use for training and evaluation.

* The second step is to define our model.

* you get a warning after instantiating this pretrained model. This is because BERT has not been pretrained on classifying pairs of sentences, so the head of the pretrained model has been discarded and a new head suitable for sequence classification has been added instead.

* The warnings indicate that some weights were not used (the ones corresponding to the dropped pretraining head) and that some others were randomly initialized (the ones for the new head).

* from transformers import Trainer  trainer = Trainer(     model,     training_args,     train_dataset=tokenized_datasets["train"],     eval_dataset=tokenized_datasets["validation"],     data_collator=data_collator,     tokenizer=tokenizer, )

* Note that when you pass the tokenizer as we did here, the default data_collator used by the Trainer will be a DataCollatorWithPadding as defined previously, so you can skip the line data_collator=data_collator in this call. It was still important to show you this part of the processing in section 2!  To fine-tune the model on our dataset, we just have to call the train() method of our Trainer:

* trainer.train()

* which should take a couple of minutes on a GPU)

* Evaluation

* compute_metrics()

* EvalPrediction

* To get some predictions from our model, we can use the Trainer.predict() command:

* The output of the predict() method is another named tuple with three fields: predictions, label_ids, and metrics.

* As you can see, predictions is a two-dimensional array with shape 408 x 2 (408 being the number of elements in the dataset we used). Those are the logits for each element of the dataset we passed to predict() (as you saw in the previous chapter, all Transformer models return logits).

* To transform them into predictions that we can compare to our labels, we need to take the index with the maximum value on the second axis:

* import numpy as np  preds = np.argmax(predictions.predictions, axis=-1)

* This time, it will report the validation loss and metrics at the end of each epoch on top of the training loss.

---
Created with [Super Simple Highlighter](https://chrome.google.com/webstore/detail/hhlhjgianpocpoppaiihmlpgcoehlhio). (C)2010-19 [Dexterous Logic](https://www.dexterouslogic.com)
