import PIL
from PIL import Image
from ultralytics import YOLO
from collections import Counter

model_yolo = YOLO("yolov8n.pt")

transform = transforms.ToTensor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def yolo_process(images, model, tresh=0.5):
    # ici images est une liste de path vers des images (ex : ["feta1.png"])
    # retourne une liste : 
    # yolo_preds_vote : liste des prédictions pour chaque image, en utilisant yolo puis le modele d'entrée 
    
    object_list = model_yolo(images, verbose=yolo_verbose)
    bxs = [o.boxes for o in object_list]
    confidences = [bx.conf for bx in bxs] # liste [image] -> liste confiance des prédictions dans l'image prédite
    crops = [bx.xyxy for bx in bxs] # liste [image] -> liste des coordonnées d'objets d'intéret dans l'image

    yolo_preds_vote = []

    for k in range(len(images2)):  # pour chq image
        ans = [] # tableau des classes prédites

        img = Image.open(images2[k]) # on ouvre l'image de base
        counted = 0

        for j in range(len(crops[k])): # pour chq objet d'intéret dans l'image
            if(confidences[k][j].item() > thresh): # si on est confiant sur la prédiction 
                new_img = img.crop((crops[k][j][0].item(),crops[k][j][1].item(),crops[k][j][2].item(),crops[k][j][3].item()))
                resized = new_img.resize((224, 224))
                resized = transform(resized).unsqueeze(0).to(device)
                model_pred = model(resized)
                ans.append(class_names[model_pred.argmax(1)])
            
        counts = Counter(ans)

        if(len(ans) != 0):
            mcs, mcc = counts.most_common(1)[0]
            yolo_preds_vote.append(mcs)
        else:
            yolo_preds_vote.append('')

    return yolo_preds_vote