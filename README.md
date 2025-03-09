# CANClassify

Implementation of CANClassify, as discussed in "CANClassify: Automated Decoding and Labeling of CAN Bus Signals"

Paul Ngo and Jonathan Sprinkle and Rahul Bhadani
EECS Department, University of California, Berkeley
Technical Report No. UCB/EECS-2022-151
May 20, 2022
Report can be found [here](http://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-151.html).
Abstract:
Controller Area Network (CAN) bus data is used on most vehicles today to report and communicate sensor data. However, this data is generally encoded and is not directly
interpretable by simply viewing the raw data on the bus. However, it is possible to decode CAN bus data and reverse engineer the encodings by leveraging knowledge about
how signals are encoded and using independently recorded ground-truth signal values for correlation. While methods exist to support the decoding of possible signals, these
methods often require additional manual work to label the function of each signal. In this paper, we present CANClassify --- a method that takes in raw CAN bus data, and
automatically decodes and labels CAN bus signals, using a novel convolutional interpretation method to preprocess CAN messages. We evaluate CANClassify's performance on a previously undecoded vehicle and confirm the encodings manually. We demonstrate performance comparable to the state of the art while also providing automated labeling. Examples and code are available at [https://github.com/ngopaul/CANClassify](https://github.com/ngopaul/CANClassify).

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

