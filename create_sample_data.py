import os
import sys
import random
import requests
import pandas as pd


URL = "https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG"
DATA_LEN = int(1e3)

if len(sys.argv) != 2:
    sys.exit("Usage: python create_sample_data.py output_dir")
out_dir = sys.argv[1]
out_path = os.path.join(out_dir, "sample.csv")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

img_path = os.path.join(out_dir, "panda.jpg")
with open(img_path, "wb") as img_file:
    raw = requests.get(URL).content
    img_file.write(raw)

data = []
for i in range(DATA_LEN):
    label = random.randint(0, 9)  # dummy label
    data.append([img_path, label])
pd.DataFrame(data, columns=["Image", "Label"]).to_csv(out_path, index=False)

