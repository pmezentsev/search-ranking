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
* ###  [Model Training Notebook](training/model_training.ipynb)

## Model Inference
The model inference part of the project is implemented using a Flask API. The search engine works in two stages:

1. Approximate candidate searching using the [FAISS (Facebook AI Similarity Search)](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)  library.
The configuration used for FAISS is "IVF800,PQ10". This configuration is a combination of two algorithms:
- **IVF800**: This part of the configuration refers to an Inverted File (IVF) index with 800 centroids. 
  The IVF index is a type of partitioning index that divides the dataset into non-overlapping clusters. 
  Each cluster is represented by a centroid, which is the center point of the cluster. The number 800 indicates 
  that the dataset is divided into 800 clusters. When searching for similar documents, only the clusters that are 
  most likely to contain similar items are searched, making the process more efficient than a brute-force search.
- **PQ10**: This part of the configuration refers to [Product Quantization (PQ)](https://www.pinecone.io/learn/product-quantization/) 
  with 10 subquantizers. Product Quantization 
  is a technique used to compress high-dimensional vectors into a more compact representation while preserving their 
  similarity information. The number 10 indicates that the PQ encoding uses 10 subquantizers. The high-dimensional 
  vectors are divided into 10 non-overlapping segments, and each segment is quantized independently. This compression 
  helps reduce the memory requirements and speeds up the search process.

Implementation details could be found here: [candidate_model.py](serving/src/candidate_model.py) 

2. Precise ranking and returning of the top N most relevant documents using the trained KNRM model. 
   (See the KNRM implementation here: [ranking_model.py](serving/src/ranking_model.py))
   

### Service API:
Below is the API description for the provided Flask application:

```
/ping (GET)
```

Example response:

```json
{ "status": "ok" }
```
---

```
/query (POST)
```
This endpoint is used to search for relevant documents based on the input queries. The request body should contain a JSON object with a key named queries, which is an array of query strings.

Example request body:

```json
{
    "queries": ["query1", "query2"]
}
```

The response is a JSON object containing the search results for each query.

Example response:

```
{
    "query1": [results for query1],
    "query2": [results for query2]
}
```
-----
```
/update_index (POST)
```

This endpoint is used to update the search engine's index with new documents. The request body should contain a JSON object with a key named documents, which is an array of document strings.

Example request body:

```
{
    "documents": ["doc1", "doc2", "doc3"]
}
```
The response is a JSON object containing a status key and the current index size after the update.

Example response:

```
{
    "status": "ok",
    "index_size": updated_index_size
}
```
-----
```
/score (POST)
```

This endpoint is used to calculate the relevance score between a document and a query. The request body should contain a JSON object with two keys: doc (the document string) and query (the query string).

Example request body:

```json
{
    "doc": "example document",
    "query": "example query"
}
```
The response is a JSON object containing the relevance score between the document and the query.

Example response:

```
{
    "score": calculated_score
}
```
