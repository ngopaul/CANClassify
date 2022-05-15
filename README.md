# CANClassify

Implementation of CANClassify.

## Setup & Requirements

- Python 3.8
- `requirements.txt` contains pip requirements. Install with 
`pip install -r requirements.txt`

## Run

Training data must be downloaded from cyverse into the `data/` folder.

The jupyter notebook files demonstrate how to 1) Preprocess Data, 2) Train 
the Model, 3) Predict Using the Model. 

In order to modify the training set and signal types to predict on, `main.py`
must be modified. Modify `labels` and any `label_to_messig` dictionaries.

