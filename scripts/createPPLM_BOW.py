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


with open("../data/amzn/train_keyword.jsonl", "r") as fp:
    p = tqdm()
    p.set_description("Reading Input File")
    while True:
        line = fp.readline()
        
        if not line:
            break
        line = json.loads(json.dumps(ast.literal_eval(line)))
        text.append(line["text"])
        p.update()
    p.close()
fp.close()


allreviews = " ".join(text)


allreviews = nltk.word_tokenize(allreviews)
allreviews = [w for w in allreviews if not w.lower() in stop_words and w.isalpha()]

cnt = Counter()
for word in allreviews:
  cnt[word] += 1
mc = cnt.most_common(300)

print(mc)

json.dump( mc, open( "out_mc_300.json", 'w' ) )

mc = [x[0] for x in mc]

print(mc)
        