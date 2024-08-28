# LocalGPT for Chatting with Financial Documents Securly

LocalGPT allows you to converse with your documents without compromising your privacy. With everything running locally, you can be assured that no data ever leaves your computer. Dive into the world of secure, local document interactions with LocalGPT.

## Environment setup

```
conda create env -f environment.yml
conda activate localGPT2
```

## UI Interface
we have created a UI from where you can find two options. First option is "Chat with LLM" where anyone can chatting with the LLM without uploading documents. It does not have any knowledge about your document and it will answer from its prior knowledge.

Another options is "Chat With Document" where you can upload any documents like PDF, excell file, txt file etc,. After uploading the file, a chatting interface will open for chatting with the document. To run the UI run the bellow command,
```
streamlit run UI.py
```

## Technical details

By selecting the right local models and the power of LangChain you can run the entire RAG pipeline locally, without any data leaving your environment, and with reasonable performance.

- **ingest.py** uses LangChain tools to parse the document and create embeddings locally using InstructorEmbeddings. It then stores the result in a local vector database using Chroma vector store. 

- **constants.py** You can change the LLM models. The formate is presented in the constants.py file

- **UI.py** file contains the user interface for the localGPT. It contains an end-to-end pipeline to run the localGPT.

