# RAG LANGCHAIN RAGAS ⛓️

![Screenshot 2024-02-08 172210](https://github.com/benitomartin/mlops-car-prices/assets/116911431/4cf4eb86-fc0e-4126-86c6-5585e72a097c)

This repository contains a full Q&A pipeline using LangChain framework, FAISS as vector database and RAGAS as evaluation metrics. The data used is the Hallucinations Leaderboard from **[HuggingFace](https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations)**. 

The notebook was run using **google colab (GPU required)**

The main steps taken to build the RAG pipeline can be summarize as follows (a basic RAG Pipeline is performed after text cleaning):

* **Model Definition**: model class definition

* **Data Ingestion**: load data from website

* **Instantiation**: model llama2-7b or falcon-7b-v2

* **Indexing**: RecursiveCharacterTextSplitter for indexing in chunks

* **Embedding**: HuggingFaceBgeEmbeddings BAAI/bge-large-en-v1.5

* **QA Chain Retrieval**: HuggingFacePipeline and RetrievalQA

* **Scoring**: top k most similar results

* **Evaluation**: TestsetGenerator from RAGAS and evaluation with faithfulness, answer_relevancy, context_recall, context_precision, and answer_correctness

## 👨‍💻 **Tech Stack**


![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## 📐 Set Up

In the initial project phase, the documents are loaded using **WebBaseLoader**and indexed. Indexing is a fundamental process for storing and organizing data from diverse sources into a vector store, a structure essential for efficient storage and retrieval. This process involves the following steps:

- Select a splitting method and its hyperparameters: we will use the **RecursiveCharacterTextSplitter**. Recursively tries to split by different characters to find one that works.

- Select the embeddings model: in our case the **BAAI/bge-large-en-v1.5** (Flag Embedding which is focused on RAG LLMs).

- Select a Vector Store: **FAISS**.

Storing text chunks along with their corresponding embedding representations, capturing the semantic meaning of the text. These embeddings facilitate easy retrieval of chunks based on their semantic similarity. 

After indexing, a QA Chain Retrieval Pipeline is set up in order to check the Q&A functioning and performance.

## 🌊 QA Chain Retrieval Pipeline

The pipeline created with HuggingFacePipeline contains the main hyperparameters of the model like top_p, max_legth, tokenizer, temperature...The vector store is then set up as similarity retriever and a prompt template is used to complete the QA chain.

```
# Set the question
question = "What are hallucinations in the context of LLM models?"

# Initializing a combined QA chain
qa_combined_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type_kwargs = {"prompt": prompt},
                    retriever=retriever,
                    callbacks=[handler],
                    return_source_documents=True)

# Initialize the query
response = qa_combined_chain({"query": question, "context": retriever})
```


The results of the chain can be seen below to the question `What are hallucinations in the context of LLM models?`
```
"Hallucinations in the context of LLMs refer to instances where the model generates content that does not align with real-world facts or the user's input. This can lead to unreliable and unfaithful language generation, which can negatively impact the effectiveness of LLMs in various applications. To address this challenge, the Hallucinations Leaderboard was created as an open effort to evaluate and compare different LLMs based on their tendency to generate hallucinated content. By providing insights into the generalization properties and limitations of these models, the leaderboard aims to support the development of more reliable and accurate language generators."
```

## 🚒 Model Evaluation
###  Similarity 🤼‍♂️
--------------


Once the pipeline is set up, the query can be done for similarity scoring. To the question `What are hallucinations in the context of LLM models?` the following output with the **similarity_search_with_score** (the lower the better) was retrieved:

```
Source: https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations, Score: 0.41722849011421204
Source: https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations, Score: 0.424437552690506
Source: https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations, Score: 0.4498262107372284
```

Once the pipeline is set up, the query can be done for similarity scoring. To the question `Which are the main characters of the book?` the following output with the **similarity_search_with_relevance_scores** (the higher the better) was retrieved:

```
Source: https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations, Score: 0.7049749053360163
Source: https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations, Score: 0.6998773283023207
Source: https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations, Score: 0.6819248360322568
```

The **score** represents the relevance or similarity measure between the query and the retrieved document. This score indicates the likelihood or degree to which the document is considered relevant to the query based on the model's understanding or learned representation of the text.

### RAGAS 📊
--------------

To evaluate the model, we used the RAGAS library. **TestsetGenerator** and open-ai models allows to create synthetic evaluation dataset for assessing the RAG pipeline. It provides a variety of test generation strategies, including:

- **simple**: Simple questions based on the documents and are more complex to derive from the provided contexts.

- **reasoning**: Questions which require reasoning based on the context. Rewrite the question in a way that enhances the need for reasoning to answer it effectively.

- **multi_context**: Questions which are generated from multiple contexts from the documents. Rephrase the question in a manner that necessitates information from multiple related sections or chunks to formulate an answer.

The testset.csv file is saved under evaluation.

<p align="center">
<img alt="Screen Shot 2024-01-05 at 9 05 56 AM" src="https://github.com/benitomartin/rag_llama_deeplake/assets/116911431/2b1b232d-233c-41c0-a843-dd45f41982b8">
</p>

RAGAS utilizes Large Language Models (LLMs) to conduct evaluations across various metrics, each addressing a specific aspect of the RAG pipeline’s performance. I used the following metrics to evaluate the model:

- **Faithfulness**: measures the factual consistency of the generated answer against the given context. It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher the better.

- **Answer Relevancy (question vs answer)**: focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information. This metric is computed using the question and the answer, with values ranging between 0 and 1, where higher scores indicate better relevancy.

- **Context Precision (retrieved context vs question)**: evaluates whether all of the ground-truth relevant items (question) present in the contexts are ranked high (measures the signal-to-noise ratio of the retrieved context). Ideally all the relevant chunks must appear at the top ranks. This metric is computed using the question and the contexts, with values ranging between 0 and 1, where higher scores indicate better precision.

- **Context Recall (retrieved context vs ground truth)**: measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. It is computed based on the ground truth and the retrieved context, and the values range between 0 and 1, with higher values indicating better performance.

- **Answer Correctness (ground truth vs answer)**: evaluates the accuracy of the generated answer when compared to the ground truth. This evaluation relies on the ground truth and the answer, with scores ranging from 0 to 1. A higher score indicates a closer alignment between the generated answer and the ground truth, signifying better correctness.

The following results show the questions 3 to 6 from the above testset, taking two simple, two reasoning and one multicontext question. The results.csv file is saved under evaluation


<p align="center">
<img alt="Screen Shot 2024-01-05 at 9 05 56 AM" src="https://github.com/benitomartin/rag_llama_deeplake/assets/116911431/beb64c9f-8913-42aa-a8fd-42db80a48a34">
</p>

We can see that for the first 2 questions (simple) we reach very good results, whereas reasoning is only outstading in answer relevancy and multicontext in context recall as well. Here are the key findings:

- **faithfulness**: the last two questions are not answered consistently or in other words, show hallucinations. However, by reading the answer of the third question (reasoning), part of it has been adressed, LLaMA2 13B Chat and Mistral 7B are the best models. But it also includes a third model and no information about the size. So the answer is not consistent and got 0 faithfulness

- **answer_relevancy**: all the questions got fairly relevant answers to the questions

- **context_precision**: does the model response aligns with the question relevant elements? By reading the last two answers, they do not align completely. The third question ask for the size of the model, which is not available in the answer, which do not allow to get a proper answer on that. This means that the retrieved context to generate the answer do not contain the relevant elements from the question. The last question seems to have addressed the issue a bit better and retrieved some relevant elements. 

- **context_recall**: does the retrieved context contains the ground truth information to answer the question? Here we point out again to the third question, where, the size of the model is not available, which might be the reason of the lower score compared to the other questions.

- **answer_correctness**: we see that most of the answers contain more information (relevant or not relevant) compared to the ground truth, which lowers the score. Even if the answer is correct like the first one in the first statement, it includes a second statement which is not in the ground truth. 

## 📈 Further Steps

Similarity scores has shown that even though the performance of the model is not bad, it can be further improved. 

RAGAS is a good method for evaluation of LLM models and shows outstanding results in answer relevancy and context recall. However, we can see some areas of improvement in the way it retrieved the information and adress the context precission and faithfulness.

Selecting other database, model, embeddings or performing hyperparameter tuning can enhance the results and further improve the metrics.

* Different database: `Deep Lake`, `Chroma`, `Pinecode`,...
* Different model: Huggingface `BAAI/bge-reranker-base` reranker model
* Hyperparameter tuning: top_p, k, chunk size,...
