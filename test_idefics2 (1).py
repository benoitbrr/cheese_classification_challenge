import PIL
from PIL import Image
import os
from IPython.display import display
import random
import numpy as np
import requests
import torch
from io import BytesIO
import copy
from fuzzywuzzy import fuzz

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda"

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

names = os.listdir(lbl_img_path)

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
     torch_dtype=torch.float16,
).to(DEVICE)

messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text":
            " Is there a label on the image ? "
            }
             ]
    },
]

messages2 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text":

              #"Is there one of the following labels : roquefort, comte, brie, maroilles, chabichou, epoisses, ossau-iraty, stilton, fourme d'ambert, mozzarella, mont d'or, camembert, cabecou, emmental, mothais, neufchatel, feta, gruyère, mimolette, parmigiano, parmesan or none of these ?",
            "Is there one of the following labels : " +
             "BRIE, ".lower() +
             "CAMEMBERT, ".lower()+
"EPOISSES, ".lower()+
"AMBERT, ".lower()+
"RACLETTE, ".lower()+
"MORBIER, ".lower()+
"NECTAIRE, ".lower()+
"POULIGNY, ".lower()+
"SAINT-PIERRE, ".lower()+
"ROQUEFORT, ".lower()+
"COMTÉ, ".lower()+
"CHÈVRE, ".lower()+
"PECORINO, ".lower()+
"NEUFCHATEL, ".lower()+
"CHEDDAR, ".lower()+
"BÛCHETTE, ".lower()+
"PARMESAN, ".lower()+
"FÉLICIEN, ".lower()+
"MONT D’OR, ".lower()+
"STILTON, ".lower()+
"SCARMOZA, ".lower()+
"CABECOU, ".lower()+
"BEAUFORT, ".lower()+
"MUNSTER, ".lower()+
"CHABICHOU, ".lower()+
"TOMME, ".lower()+
"VACHE, ".lower()+
"REBLOCHON, ".lower()+
"EMMENTAL, ".lower()+
"FETA, ".lower()+
"OSSAU, ".lower()+
"IRATY, ".lower()+
"MIMOLETTE, ".lower()+
"MAROILLES, ".lower()+
"GRUYÈRE, ".lower()+
"MOTHAIS, ".lower()+
"VACHERIN, ".lower()+
"MOZZARELLA, ".lower()+
"MOINES, ".lower()+
"FROMAGE FRAIS, " .lower()+
             "or none of these ?"
             }
             ]
    },
]

messages3 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text":
                "Do you see the following label : "
            }
             ]
    },
]

def ans(message, image):
  prompt = processor.apply_chat_template(message, add_generation_prompt=True)
  inputs = processor(text=prompt, images=[image], return_tensors="pt")
  inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

  generated_ids = model.generate(**inputs, max_new_tokens=50)
  generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

  word = '\nAssistant:'
  index = generated_texts[0].find(word)
  return generated_texts[0][index + len(word) : index + len(word) + 15]


def process(message1, message2, message3, image):
  ans_q1 = ans(message1, image)
  if "Yes" in ans_q1:
    ans_q2 = ans(message2, image)
    if "None" in ans_q2 or "No" in ans_q2 or "no" in ans_q2 :
      return ""
    else:
      message4 = copy.deepcopy(message3)
      message4[0]["content"][1]["text"] += ans_q2[:-1] + " on this image ? "
      ans_q3 = ans(message4, image)
      if "Yes" in ans_q3 :
        return ans_q2
      else :
        return ""
  else:
    return ""


def string_distance(str1, str2):
    return fuzz.ratio(str1, str2)

def acc(thresh):
    acc = 0.
    for i in range(len(names)):
        file_path = os.path.join(lbl_img_path, names[i])
        image = Image.open(file_path)
        predicted_class = process(messages1, messages2, messages3, image)
        if(predicted_class == ""):
            acc += 1.
        else:
            predicted_class = predicted_class[1:-1].lower()
            # on récupère la classe correspondante
            if(string_distance(objs[i], predicted_class) > thresh):
                acc += 1.
    return acc / len(names)

max_acc = 0.
for i in range(10):
    cur_acc = acc(70 + 2*i)
    if(cur_acc > max_acc):
        max_acc = cur_acc

print(max_acc)

        



  

        



