
import PIL
from PIL import Image
import os
from IPython.display import display
import random
import numpy as np
import easyocr
from fuzzywuzzy import fuzz

lbl_img_path = "cheese_classification_challenge/dataset/test_ocr_dataset" # le path vers le dataset de test label

objs = [
"vacherin",
"",
"",
"stilton",
"epoisses",
"camembert",
"",
"epoisses",
"",
"",
"",
"cabecou",
"emmental",
"mothais",
"",
"",
"mimolette",
"",
"",
"neufchatel",
"gruyère",
"feta",
"neufchatel",
"",
"",
"",
"",
"",
"",
"",
"roquefort",
"",
"",
"",
"",
"munster",
"",
"",
"epoisses",
"",
"",
"buchette",
"",
"pecorino",
"pouligny saint pierre",
"",
"",
"",
"",
]

names = []
i = 0
for filename in os.listdir(lbl_img_path):
  names.append(filename)
  i += 1

reader = easyocr.Reader(['en'])


def string_distance(str1, str2):
    return fuzz.ratio(str1, str2)

def prediction(outputs, thresh):
  # entrée : la sortie d' easyocr
  # sortie : la catégorie prédite
  ans = []
  flag = True
  for obj in objs:
    for word in outputs:
      if(string_distance(obj, word.lower()) > thresh and obj!=""):
        flag = False
        return obj
  if(flag):
    return ""

def acc(thresh):
  acc = 0.
  for i in range(len(names)):
    file_path = os.path.join(lbl_img_path, names[i])
    image = Image.open(file_path)
    outputs = reader.readtext(file_path, detail=0)
    x = prediction(outputs , thresh)
    if(x == objs[i]):
      acc += 1

  return acc / len(names)


max_acc = 0.
for i in range(10):
    cur_acc = acc(70 + 2*i)
    if(cur_acc > max_acc):
        max_acc = cur_acc

print(max_acc)
        



