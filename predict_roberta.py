from transformers import pipeline
from transformers import AutoModelForSequenceClassification, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoConfig, RobertaTokenizer
from scipy.special import softmax
import pandas as pd
import numpy as np
import torch
import tensorflow as tf

# Load Dataset
file_path = 'AllReviews.xlsx'
df = pd.read_excel(file_path)

# Config
config = AutoConfig.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
# Use base tokenizer
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
# Use the sentiment model
model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment") 


def hf_predict(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded_input)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)
    predicted_class = np.argmax(scores)
    if(predicted_class == 0):
        return predicted_class
    elif(predicted_class == 1):
        predicted_class == int(scores[1])
        return predicted_class
    else:
        return predicted_class  # 0=negative, 1=neutral, 2=positive

df['hf_sentiment'] = df['review_text'].apply(hf_predict)
df.head(n=20)