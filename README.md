# Cheese Classification challenge
This codebase is the code we used for the INF473V challenge.

## Instalation

Cloning the repo:
```
git clone git@github.com:benoitbrr/cheese_classification_challenge.git
cd cheese_classification_challenge
```
Install dependencies:
```
conda create -n cheese_challenge python=3.10
conda activate cheese_challenge
pip install -r requirements.txt
```
## Using this codebase
This codebase is centered in the file Implement.ipynb
You just have to run the code (update the path before that) in order to train the model, and then to create a submission

### Training

To train your model you can run 

```
python train.py
```

This will save a checkpoint in checkpoints with the name of the experiment you have. Careful, if you use the same exp name it will get overwritten

to change experiment name, you can do

```
python train.py experiment_name=new_experiment_name
```
## Create submition
To create a submition file, you can run 
```
python create_submitio_with_ocr.py experiment_name="name_of_the_exp_you_want_to_score" model=config_of_the_exp
```

Make sure to specify the name of the checkpoint you want to score and to have the right model config
