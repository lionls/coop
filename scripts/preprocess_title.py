import gzip
import json
import re
import string
import unicodedata
from collections import defaultdict
from pathlib import Path
import pandas as pd
import click
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import ast
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from collections import Counter
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
is_noun = lambda pos: pos[:2] == 'NN' or pos[:2] == 'NNS'

def extract_KeywordTitle(rev, title):
    try:    
        reviews = " ".join(rev)
        reviews = nltk.word_tokenize(reviews)
        reviews = [w for w in reviews if not w.lower() in stop_words and w.isalpha()]
        nouns = [word for (word, pos) in nltk.pos_tag(reviews) if is_noun(pos)] 
        cnt = Counter()
        for word in nouns:
            cnt[word] += 1
        mc = cnt.most_common(15)
        mc = [x[0] for x in mc]
        title = nltk.word_tokenize(title)
        return " ".join([value for value in title if value.lower() in mc])
    except:
        return "NaN"

df = pd.read_csv("../data/asintitle.csv")
df.set_index(["asin"])



def get_title(asin:str):
    try: 
        return df.loc[df["asin"] == asin, "title"].values.item()
    except:
        return "NaN"

curr = "159985130X"
lines = []

def add(x,keywords):
    x["keyword"] = keywords
    return json.dumps(x) + "\n"

def write_lines(fout, lines, keywords):
    lines = [add(x,keywords) for x in lines]
    fout.writelines(lines)


with open("../data/amzn/train_title.jsonl", "w") as fout:

    with open("../data/amzn/train.jsonl", "r") as fp:
        p = tqdm()
        while True:
            line = fp.readline()
            
            if not line:
                break
                
            line = json.loads(json.dumps(ast.literal_eval(line)))
            if curr != line["business_id"]:
                title = get_title(curr)
                rev = [str(x["text"]) for x in lines]
                keywords = extract_KeywordTitle(rev,title)
                print(keywords)
                write_lines(fout, lines, keywords)
                lines = []
                curr = line["business_id"]
                p.update()
            lines.append(line)

        p.close()
        fp.close()
    fout.close()
            