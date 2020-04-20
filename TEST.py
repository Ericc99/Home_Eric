#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_train = pd.read_csv("./train.csv")
print(df_train.head())


# In[ ]:


from torchtext import *
from torchtext.data import *

import nltk
nltk.download('punkt')
from nltk import word_tokenize

txt_field = data.Field(tokenize=word_tokenize, lower=True, include_lengths=True, batch_first=True)
label_field = data.Field(sequential=False, use_vocab=False, batch_first=True)

# make splits for data
train, test= TabularDataset.splits(path='./', train='train.csv', test='test.csv',format='csv', 
                                  fields=[('label', label_field),('text', txt_field)], skip_header=True)

# build the vocabulary on the training set only
txt_field.build_vocab(train, min_freq=5)
#label_field.build_vocab(train)

# make iterator for splits
train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=30, 
                                                   sort_key=lambda x: len(x.text),sort_within_batch=True)


# In[ ]:


print(f'Number of training samples: {len(train.examples)}')
print(f'Number of testing samples: {len(test.examples)}')

print(f'Example of training data:\n {vars(train.examples[0])}\n')
print(f'Example of testing data:\n {vars(test.examples[1])}\n')

_, batch = next(enumerate(train_iter))
print('label tensor', batch.label.shape)
print(batch.label)
print()
sent, sent_len = batch.text
print('sentence length tensor', sent_len.shape)
print(sent_len)
print()
print('sentence tensor', sent.shape)
print(sent)


# In[ ]:




