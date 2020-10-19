import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
nltk.download('punkt')


def read_file(file_path):
    """
    Read function for AG NEWS Dataset
    """
    data = pd.read_csv(file_path, names=["class", "title", "description"])
    texts = list(data['title'].values + ' ' + data['description'].values)
    texts = [word_tokenize(preprocess_text(sentence)) for sentence in texts]
    labels = [label-1 for label in list(data['class'].values)]  # label : 1~4  -> label : 0~3
    return texts, labels


def preprocess_text(string):
    """
    reference : https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def metrics(dataloader, losses, correct, y_hats, targets):
    avg_loss = losses / len(dataloader)
    accuracy = correct / len(dataloader.dataset) * 100
    precision = precision_score(targets, y_hats, average='macro')
    recall = recall_score(targets, y_hats, average='macro')
    f1 = f1_score(targets, y_hats, average='macro')
    cm = confusion_matrix(targets, y_hats)
    return avg_loss, accuracy, precision, recall, f1, cm