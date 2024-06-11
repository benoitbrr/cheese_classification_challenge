import torch
import wandb
import hydra
from tqdm import tqdm
from ultralytics import YOLO
from collections import Counter
import torchvision.transforms as transforms
from PIL import Image
import os


train_or_test = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_activated = False
saving = True


model_yolo = YOLO("yolov8n.pt")
transform = transforms.ToTensor()


@hydra.main(config_path="configs/train", config_name="config")
def train(cfg):
    logger = wandb.init(project="challenge_cheese", name=cfg.experiment_name)
    
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()

    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(train_loader):
            images, labels = batch
            if(epoch == 0 and i == 1):
              print("Batch_size = ", len(images))
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
            logger.log({"loss": loss.detach().cpu().numpy()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )
        val_metrics = {}
        for val_set_name, val_loader in val_loaders.items():
            epoch_loss = 0
            epoch_num_correct = 0
            num_samples = 0
            y_true = []
            y_pred = []
            for i, batch in enumerate(val_loader):
                images, labels = batch
                if(epoch == 0 and i == 1):
                  print("Batch_size = ", len(images))
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = loss_fn(preds, labels)
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(preds.argmax(1).detach().cpu().tolist())
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (
                    (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                )
                num_samples += len(images)
            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            val_metrics[f"{val_set_name}/loss"] = epoch_loss
            val_metrics[f"{val_set_name}/acc"] = epoch_acc
            val_metrics[f"{val_set_name}/confusion_matrix"] = (
                wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=[
                        datamodule.idx_to_class[i][:10].lower()
                        for i in range(len(datamodule.idx_to_class))
                    ],
                )
            )

        logger.log(
            {
                "epoch": epoch,
                **val_metrics,
            }
        )
    if(saving):
      torch.save(model.state_dict(), cfg.checkpoint_path)

@hydra.main(config_path="configs/train", config_name="config")
def test_on_val(cfg):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = hydra.utils.instantiate(cfg.model.instance).to(device) 
  checkpoint = torch.load(cfg.checkpoint_path)
  print("Modele : ", cfg.checkpoint_path)
  model.load_state_dict(checkpoint)

  class_names = sorted(os.listdir(cfg.dataset.train_path))

  datamodule = hydra.utils.instantiate(cfg.datamodule)
  val_loaders = datamodule.val_dataloader()

  transform = val_loaders["transform"]

  path = val_loaders["path"]

  print("Chemin : ", path)

  folders = os.listdir(path)
  paths = [os.path.join(path, folder) for folder in folders]

  total_acc_std = 0.
  total_acc_yolo = 0.
  total_files = 0

  for i in range(len(paths)):

    if(i > 10):
      break

    true_pred = folders[i]
    class_acc_std = 0.
    class_acc_yolo = 0.
    num_pred_yolo = 0

    for img_name in os.listdir(paths[i]):
      if(img_name.endswith(".jpg")):
        img_path = os.path.join(paths[i], img_name)

        img = Image.open(img_path)

        # Partie YOLO
        bxs = model_yolo(img_path, verbose=False)[0].boxes
        conf = bxs.conf # liste de confidences 
        crops = bxs.xyxy # liste de crops
        yolo_ans = []
        for j in range(len(crops)):
          new_img = img.crop((crops[j][0].item(),crops[j][1].item(),crops[j][2].item(),crops[j][3].item()))
          new_img = transform(new_img)
          new_img = new_img.unsqueeze(0).to(device)
          pred = model(new_img)
          yolo_ans.append(class_names[pred.argmax(1).item()])
      
        counts = Counter(yolo_ans)
        if(len(counts) == 1):
          c = counts.most_common(1)
          mcs, mcc = c[0]
        elif(len(counts) > 1):
          c = counts.most_common(2)
          mcs1, mcc1 = c[0]
          _, mcc2 = c[1]
          if(mcc1 > mcc2):
            mcs = mcs1
            num_pred_yolo += 1
          else:
            mcs = ''
        else:
          mcs = ''
      
        # Partie standard  
        img = transform(img)
        img = img.unsqueeze(0).to(device)
        pred = model(img)
        pred = pred.argmax(1).item()
        pred = class_names[pred]

        if(mcs != ''):
          yolo_pred = mcs
        else :
          yolo_pred = pred


        if(pred == true_pred):
          class_acc_std += 1.
          total_acc_std += 1.
        if(yolo_pred == true_pred):
          class_acc_yolo += 1.
          total_acc_yolo += 1.


    class_acc_std /= len(os.listdir(paths[i]))
    class_acc_yolo /=  len(os.listdir(paths[i]))
    total_files += len(os.listdir(paths[i]))

    print("Accuracy standard sur la classe : ", true_pred, " : ", class_acc_std)
    print("Accuracy YOLO sur la classe : ", true_pred, " : ", class_acc_yolo)
    print('\n')

  total_acc_std /= total_files
  total_acc_yolo /= total_files
  print("Accuracy std sur le val : ", total_acc_std)
  print("Accuracy yolo sur le val : ", total_acc_yolo)



if __name__ == "__main__":
    if(train_or_test):
      print("La phase d'entrainement est lancée")
      train()
    else:
      print("La phase de test sur le validation set est lancée")
      test_on_val()
