# KNRM based Search Engine

This project was created as the final project of the first module in the [HardML course](https://karpov.courses/ml-hard). The goal of this project is to demonstrate the development of a search engine that indexes documents and provides the most similar documents for a given query using a Flask API.

The project is divided into two main parts: model training and model inference.

## Model Training
In the model training part, we train a fine model to estimate the relevance of documents. 
This model uses the [Kernelized Neural Ranking Model (KNRM)](https://arxiv.org/abs/1706.06613) to predict 
the relevance of documents. The KNRM model is trained on the 
[Quora question similarity dataset](https://paperswithcode.com/dataset/quora-question-pairs), 
which consists of pairs of questions with binary labels indicating whether the questions are 
duplicates of each other.

We measured the quality of the model using the 
[NDCG (Normalized Discounted Cumulative Gain)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
metric, which is a widely used measure of ranking performance. The trained model achieved an 
NDCG score of **0.94047**, indicating a high level of accuracy in its ranking predictions.

To go over steps and train your own model relevance model, clone the project use the
* ###  [Model Training Notebook](training/model_training.ipynb).

####Model Inference
The model inference part of the project is implemented using a Flask API. The search engine works in two stages:

Approximate candidate searching using the [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) (Facebook AI Similarity Search) library.
FAISS library configuration: (insert FAISS library configuration details here)
Precise ranking and returning of the top N most relevant documents using the trained KNRM model.
The final quality metrics for the search engine are:

(insert final quality metrics here)
Key Points
This project was completed as part of the HardML course.
It serves as an example of a search engine that indexes documents and provides the most similar documents for a given query using a Flask API.
The project is divided into two main parts: model training and model inference.
The trained model uses the KNRM for relevance prediction and is trained on the Quora question similarity dataset.
The model inference part is implemented using a Flask API, and the search engine works in two stages: approximate candidate searching using the FAISS library and precise ranking using the KNRM model.
The final quality metrics for both the model and the search engine are provided.
To get started with the project, follow the instructions provided in the respective Jupyter Notebook (for model training) and the Flask API documentation (for model inference).



# How to run this project:

how to train the model
how to build docker file


how to run it locally


endpoints desciption

  
   
