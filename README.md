# POC detection
Machine learning model for detecting pockets of open cells (POCs). It consists 
of two components: a 'rough' model and a second 'refined' model. 

Model description and current best accuracy can be found in the 
paper: https://doi.org/10.1029/2020GL092213

## Training data

The hand-labelled tracks and images used for training can be found here: https://zenodo.org/record/6810590


## Running the model

Simply run the `poc_model.py` script by providing a numpy file of the images
and the path to the model weights.

The weights used for the model we ran in the paper can be found here: https://zenodo.org/record/6810590
