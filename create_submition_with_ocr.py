import hydra
from torch.utils.data import Dataset, DataLoader
import os
import PIL
from PIL import Image
import pandas as pd
import torch
from IPython.display import display
import torchvision.transforms as transforms
from os import listdir
from os.path import isfile, join
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='cheese_classification_challenge/aesthetic-frame-414017-bb70db4249a8.json'
from google.cloud import vision
import time

import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------------------




cheeses = {
"BRIE DE MELUN": ["Brie", "BRIE", "brie", "MELUN", "Melun", "melun"],
"CAMEMBERT":["Camembert", "CAMEMBERT", "camembert"],
"EPOISSES":["EPOISSES", "Epoisses", "epoisses"],
"FOURME D’AMBERT":["Fourme", "FOURME", "fourme", "Ambert", "ambert", "AMBERT"],
"RACLETTE":["RACLETTE", "Raclette", "raclette"],
"MORBIER":["MORBIER", "Morbier", "morbier"],
"SAINT-NECTAIRE":["nectaire", "NECTAIRE", "Nectaire"],
"POULIGNY SAINT- PIERRE":["Pouligny", "POULIGNY", "pouligny"],
"ROQUEFORT":["roquefort", "ROQUEFORT", "Roquefort"],
"COMTÉ":["Comté", "COMTÉ", "comté", "comte","Comte", "COMTE"],
"CHÈVRE":["Chèvre", "CHEVRE", "chèvre", "chevre","Chevre", "CHÈVRE"],
"PECORINO":["Pecorino", "pecorino", "PECORINO"],
"NEUFCHATEL":["NEUFCHATEL", "neufchatel", "Neufchatel"],
"CHEDDAR":["cheddar", "Cheddar", "CHEDDAR"],
"BÛCHETTE DE CHÈVRE":["buchette", "BUCHETTE", "Buchette","bûchette", "BÛCHETTE", "Bûchette" ],
"PARMESAN":["parmesan", "PARMESAN", "Parmesan"],
"SAINT- FÉLICIEN":["félicien", "Félicien", "FÉLICIEN", "felicien", "Felicien", "FELICIEN"],
"MONT D’OR":["mont", "MONT", "Mont"],
"STILTON":["Stilton", "STILTON", "Stilton"],
"SCARMOZA":["SCARMOZA", "Scarmoza", "scarmoza"],
"CABECOU":["Cabecou", "cabecou", "CABECOU"],
"BEAUFORT":["BEAUFORT", "Beaufort", "beaufort"],
"MUNSTER":["MUNSTER", "Munster", "munster"],
"CHABICHOU":["CHABICHOU", "Chabichou", "chabichou"],
"TOMME DE VACHE":["tomme", "TOMME", "Tomme", "VACHE", "Vache", "vache"],
"REBLOCHON":["reblochon", "REBLOCHON", "Reblochon"],
"EMMENTAL":["emmental", "Emmental", "EMMENTAL"],
"FETA":["Feta", "FETA", "feta"],
"OSSAU- IRATY":["Ossau", "OSSAU", "ossau", "IRATY", "Iraty", "iraty"],
"MIMOLETTE":["MIMOLETTE", "mimolette", "Mimolette"],
"MAROILLES":["Maroille", "MAROILLE", "maroille"],
"GRUYÈRE":["gruyère", "GRUYÈRE", "Gruyère","gruyere", "GRUYERE", "Gruyere"],
"MOTHAIS":["MOTHAIS", "Mothais", "mothais"],
"VACHERIN":["vacherin", "VACHERIN", "Vacherin"],
"MOZZARELLA":["Mozzarella", "MOZZARELLA", "mozzarella"],
"TÊTE DE MOINES":["moines", "MOINES", "Moines"],
"FROMAGE FRAIS": ["Fromage frais", "Fromage Frais", "fromage frais", "fromage Frais", "FROMAGE FRAIS"]
}



# prend en entrée un path vers une image et retourne une liste de mots detectés sur l'image 
def detect_text(path):
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as image_file:
        content = image_file.read()
    #content = Image.open(path)
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    ocr_text = []
    for text in texts[1:]:
        ocr_text.append(text.description)
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    if(texts != []):
      return ocr_text
    else:
      return [""]

# entrée : liste de mots 
# sortie : booléen label ? et classe prédite si booléen label est vrai
def word_to_class(words):
  ret_bool = False
  ret_class = None

  for i in range(len(words)):
    for keys in cheeses:
      if(words[i] in cheeses[keys] and not ret_bool):
        ret_class = keys
        ret_bool = True
        break

  return ret_bool, ret_class

def to_list(path_list):
  flags = []
  preds = []
  for path in path_list:
    words_detected = detect_text(path)
    b, c = word_to_class(words_detected)
    flags.append(b)
    preds.append(c)
  return flags, preds


transform = transforms.ToTensor()

# PARAMS : 

using_model = True
borne_inf = 0.5
freq = 15
freq_batch = 1
yolo_verbose = False
breaker = False
max_iter = 25
ratio_img = 0.8
printer = True

# ---------------------------------------------------------------------------------------


class TestDataset(Dataset):
    def _init_(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def _getitem_(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def _len_(self):
        return len(self.images_list)


@hydra.main(config_path="configs/train", config_name="config")
def create_submission(cfg):

    test_loader = DataLoader(
        TestDataset(
            cfg.dataset.test_path, hydra.utils.instantiate(cfg.dataset.test_transform)
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    # Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    if(using_model):
      checkpoint = torch.load(cfg.checkpoint_path)
      print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
      model.load_state_dict(checkpoint)
    class_names = sorted(os.listdir(cfg.dataset.train_path))

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])

    for i, batch in enumerate(test_loader):
        if(i % freq_batch == 0):
          images, image_names = batch
          images = images.to(device)

          # type(images) : torch.tensor
          # shape(images) : batch_size, 3, 224, 224

          # -------------------------

          # images2 est la liste des path vers les images du batch
          images2 = []
          for names in image_names:
            labeled_images_path = os.path.join(cfg.dataset.test_path, names)
            images2.append(labeled_images_path + '.jpg')

          bls, classes = to_list(images2)

          #print("Ici : ")
          preds = model(images)
          preds_confs = preds.max(1).values
          preds = preds.argmax(1)
          preds = [class_names[pred] for pred in preds.cpu().numpy()]

          nb_im_label = 0.
          nb_im_same_pred = 0.
          nb_yolo_modified = 0.

          for j in range(len(bls)):
            if(bls[j]):
              nb_im_label += 1.
              if(preds[j] == classes[j]):
                nb_im_same_pred += 1.
              preds[j] = classes[j]

          ratio_same_pred  =  100 * nb_im_same_pred / len(bls)
          ratio_label = 100 * nb_im_label / len(bls)
          ratio_same_yolo = 100 * nb_yolo_modified / len(bls)

          if(i % freq == 0 and printer):
            print("Dans ce batch, " + str(ratio_label) + "% des images ont un label selon notre modèle d'OCR")
            print("Dans ce batch, " + str(ratio_same_pred) + "% des images ont la même prédiction par le modèle que par l'OCR")
            print("Dans ce batch, " + str(ratio_same_yolo) + "% des images ont une prédiction qui vient de YOLO")
            print('\n')

          # -------------------------

          if(breaker):
            if(i > max_iter):
              break

          #preds = model(images)
          #preds = preds.argmax(1)
          #preds = [class_names[pred] for pred in preds.cpu().numpy()]

          submission = pd.concat(
              [
                  submission,
                  pd.DataFrame({"id": image_names, "label": preds}),
              ]
          )
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)

if _name_ == "_main_":
    print('\n')
    print("Lors de cet entrainement, voici la valeur des paramètres :")
    print("using_model : " +  str(using_model))
    print("borne_inf : " + str(borne_inf))
    print("freq : " + str(freq))
    print("verbose : " + str(yolo_verbose))
    print("breaker : " + str(breaker))
    print("max_iter : " + str(max_iter))
    print('\n')
    create_submission()
