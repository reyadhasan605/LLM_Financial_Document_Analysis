# FinancialGPT for Chatting with Financial Documents Securely

FinancialGPT allows you to converse with your documents without compromising your privacy. With everything running locally, you can be assured that no data ever leaves your computer. Dive into the world of secure, local document interactions with FinancialGPT.

## Environment setup

```
conda create env -f environment.yml
conda activate localGPT2
```

## UI Interface
we have created a UI from where you can find two options. The first option is "Chat with LLM" where anyone can chat with the LLM without uploading documents. It does not have any knowledge about your document and it will answer from its prior knowledge.

Another option is "Chat With Document" where you can upload any document like PDF, excel file, txt files, etc, After uploading the file, a chat interface will open for chatting with the document. To run the UI run the below command,
```
streamlit run UI.py
```

## Technical details

By selecting the right local models and the power of LangChain you can run the entire RAG pipeline locally, without any data leaving your environment, and with reasonable performance.

- **ingest.py** uses LangChain tools to parse the document and create embeddings locally using InstructorEmbeddings. It then stores the result in a local vector database using Chroma vector store. 

- **constants.py** You can change the LLM models. The format is presented in the constants.py file

- **UI.py** file contains the user interface for the FinancialGPT. It contains an end-to-end pipeline to run the FinancialGPT.

