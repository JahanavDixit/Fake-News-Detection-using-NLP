# Fake News Detection using NLP

This repository contains code for detecting fake news using natural language processing (NLP). The model is trained using BERT, a pre-trained transformer-based model for natural language processing.

## Dataset
The model is trained on a dataset of news articles. The dataset contains both real and fake news articles. The dataset used for training is [Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) dataset, which is a publicly available dataset for fake news detection.

## Model Architecture
The model architecture is based on BERT, which is a pre-trained transformer-based model. The model is fine-tuned on the fake news detection dataset using Tensorflow.

## Usage
To use this model, you can run the Jupyter notebook in Google Colab or on your local machine. The notebook contains code for preprocessing the dataset, training the model, and testing the model on new data.

### Requirements
- Python 3.x
- Tensorflow , Keras
- Transformers
- Pandas
- Numpy

### Instructions
1. Clone this repository.
2. Install the required packages using pip.
3. Open the Jupyter notebook `Fake_News_Detection.ipynb` in Google Colab or on your local machine.
4. Run the notebook cells to preprocess the dataset, train the model, and test the model on new data.

## Results
The model achieved an accuracy of 91-93% on the test dataset.

## Credits
The dataset used for training the model is [Fake News dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset), which is publicly available.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

