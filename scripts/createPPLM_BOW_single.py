import gzip
import json
import re
import string
import unicodedata
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import ast
import nltk
from collections import Counter
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))

text = []
PRINTABLE = set(string.printable)

MIN_REV_LEN = 4
MAX_REV_LEN = 128

dir_path="../data/amzn/data/amazon"

def strip_text(s: str) -> str:
    # https://stackoverflow.com/a/518232/2809427
    # https://stackoverflow.com/a/8689826
    return re.sub(" +", " ", "".join(c for c in unicodedata.normalize("NFD", s)
                                     if unicodedata.category(c) != "Mn" and c in PRINTABLE).replace("\n", " "))

def createmap(text,fp):
    allreviews = " ".join(text)
    allreviews = nltk.word_tokenize(allreviews)
    allreviews = [w for w in allreviews if not w.lower() in stop_words and w.isalpha()]
    cnt = Counter()
    for word in allreviews:
        cnt[word] += 1
    mc = cnt.most_common(300)
    print(mc)
    json.dump(mc, open( fp, 'w' ) )

p = tqdm()

for fp in Path(dir_path).glob("*.gz"):
    p.set_description(desc=fp.stem)
    d = defaultdict(list)
    for ins in map(json.loads, gzip.open(fp, "rb")):
        textrev = strip_text(ins["reviewText"])
        rating = int(float(ins["overall"]))
        review_id = ins["reviewerID"]
        text.append(textrev)
        p.update()

    createmap(text,fp)
p.close()
