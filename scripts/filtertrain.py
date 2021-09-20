import gzip
import json
import re
import string
import unicodedata
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import ast

with open("../data/amzn/train_title.jsonl", "r") as fp:

    with open("../data/amzn/train_filtered.jsonl", "w") as fout:
        p = tqdm()
        while True:
            line = fp.readline()
            
            if not line:
                break
                
            line = json.loads(json.dumps(ast.literal_eval(line)))
            k = line["keyword"]
            if(k == "" or k == "NaNkey"):
                continue
            
            fout.writelines([json.dumps(line)+"\n"])
            p.update()
            

        p.close()
        fout.close()
    fp.close()
            