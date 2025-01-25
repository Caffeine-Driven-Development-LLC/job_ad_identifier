# Job Ad Identifier

This project trains a deep learning model to identify a job ad from a url,

This project requires python 3.11

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/extract_and_split_data.py
TF_USE_LEGACY_KERAS=1 python src/train_model.py
python src/test_model.py
```

each of these scripts should be run in order,

The first will strip relevant data from a sqlite database and split it into training and testing data. This will also generate the word_index that will be used when making predictions.

The train model script trains a tensorflow model on the training data and saves the model to disk.

The test model script loads the model and tests it on the test data to get an idea of how accurate the model is.

Currently, the model is running at 99.2% accuracy on the test data.

