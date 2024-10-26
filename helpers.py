import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
import string
import pickle
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments



nlp = spacy.load("en_core_web_lg")
vectorizer = pickle.load(open('./vectorizer.sav','rb'))

# Load the fine-tuned model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('./roberta-tokenizer-2')
model = RobertaForSequenceClassification.from_pretrained('./roberta-model-2')

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def countCapitalWords(sentence:str):
    return sum(1 for word in sentence.split(" ") if word and (64< ord(word[0]) < 91))

def countCapitalLetters(sentence:str):
    return sum(1 for word in sentence.split(" ") for ch in word if 'A' <= ch <= 'Z')

def countPunctuations(sentence:str):
    punctuation = set(string.punctuation)
    return sum(1 for word in sentence.split(" ") for ch in word if ch in punctuation)

def countWords(sentence:str):
    return len(sentence.split(" "))


def extract_pos(combined_df):
    ## https://universaldependencies.org/u/pos/
    pos_tags = {
        "ADJ": [],
        "ADP": [],
        "ADV": [],
        "AUX": [],
        "CCONJ": [],
        "DET": [],
        "INTJ": [],
        "NOUN": [],
        "NUM": [],
        "PART": [],
        "PRON": [],
        "PROPN": [],
        "PUNCT": [],
        "SCONJ": [],
        "SYM": [],
        "VERB": [],
    }

    def posExtractor(sentence):
        hashmap = Counter([word.pos_ for word in nlp(sentence)])
        for k in pos_tags.keys():
            if k in hashmap:
                pos_tags[k].append(hashmap[k])
            else:
                pos_tags[k].append(0)

    for sentence in tqdm(combined_df["text"]):
        posExtractor(sentence)
    return pos_tags


def find_tense_count(sentences):
    doc = nlp(sentences)

    # Function to identify tense
    def detect_tense(token):
        if token.tag_ == "VBG" and any(aux.lemma_ in ["be", "am", "is", "are", "was", "were"] for aux in token.head.children):
            return "continuous"
        elif token.tag_ == "VBN" and any(aux.lemma_ in ["have", "has", "had"] for aux in token.head.children):
            return "perfect"
        elif token.tag_ == "VB" and any(aux.lemma_ == "will" for aux in token.head.children):
            return "future"
        elif token.tag_ in ["VBD", "VBP", "VBZ"]:
            if token.dep_ == "ROOT" and any(aux.dep_ == "aux" and aux.lemma_ == "do" for aux in token.children):
                return "simple present"
            else:
                return "simple past"
        return "unknown"

    # Detect tenses in the doc
    tense_counts = {"continuous": 0, "perfect": 0, "future": 0, "simple present": 0, "simple past": 0}

    for token in doc:
        tense = detect_tense(token)
        if tense in tense_counts:
            tense_counts[tense] += 1

    return tense_counts.values()


def makeTestingDataStats(sentences):
    df = pd.DataFrame({"text":[sentences]})
    for k,v in extract_pos(df).items():
        df[k] = v
    df["word_counts"] = df["text"].apply(countWords)
    df["capital_words"] = df["text"].apply(countCapitalWords)
    df["capital_letters"] = df["text"].apply(countCapitalLetters)
    df["punctuations"] = df["text"].apply(countPunctuations)
    df[['continuous', 'perfect', 'future', 'simple present', 'simple past']] = df["text"].apply(find_tense_count).apply(pd.Series)
    return df[df.columns[1:]]


def makeTestingDataVect(df):
    return vectorizer.transform([df])


def preprocess_input(sentences):
    # Tokenize the input sentences
    encoding = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # Move tensors to the same device as the model
    encoding = {key: value.to(device) for key, value in encoding.items()}
    return encoding

def predict(sentences):
    # Preprocess the input sentences
    inputs = preprocess_input(sentences)

    # Put the model in evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to get probabilities
    probs = torch.softmax(outputs.logits, dim=-1)

    # Get the predicted class (the one with the highest probability)
    predicted_class = torch.argmax(probs, dim=-1).cpu().numpy()

    return probs

def classify_user_input(sentence):

    # Get predictions
    predictions = predict(sentence)
    return predictions[0]