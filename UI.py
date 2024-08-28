#################  For model #####################
import time
import os
import re
import logging
import click
import torch
import utils
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
import shutil
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template
from utils import get_embeddings

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
import streamlit as st
import os
import subprocess
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
)



def load_model(device_type, model_id, model_basename=None, LOGGING=logging):

    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):


    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa

def main(device_type="cuda" , show_sources=False, use_history=False, model_type='llama'):

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)

    return qa






def clear_directory():
    directory = 'SOURCE_DOCUMENTS'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                # Remove the file
                os.unlink(file_path)
            # Check if the path is a directory
            elif os.path.isdir(file_path):
                # Remove the directory and its contents recursively
                os.rmdir(file_path)
        except Exception as e:
            not_found_files = st.empty()
            not_found_files.success(f'Something wrong')
            time.sleep(2)
            not_found_files.empty()

    #### clear DB directory #######

    not_found_files = st.empty()
    directory = 'DB'
    try:
        # Attempt to remove the directory and its contents
        shutil.rmtree(directory)
    except OSError as e:
        not_found_files.success(f'Something wrong')
        time.sleep(2)
        not_found_files.empty()


def process_file():
    st.sidebar.warning('File uploading. This may take some time.')
    subprocess.run(['python','ingest.py'])
    st.sidebar.success('File uploaded successfully')

st.error('Select the options from sidebar')
st.title("Chat With your Financial Documents Securly")

options = st.sidebar.selectbox("Select options",("Chat with LLM","Chat with document"))
qa = main()

if "messages" not in st.session_state:
    st.session_state["messages"] = []


for chat in st.session_state.messages:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])


if options is "Chat with document":
    uploaded_file = st.sidebar.file_uploader(label="Select file")
    if uploaded_file is not None:
        os.makedirs('SOURCE_DOCUMENTS', exist_ok=True)
        with open(os.path.join('SOURCE_DOCUMENTS', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())

        if st.sidebar.button('Upload files'):
            process_file()

    query = st.text_input("Questions: ")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        if query == 'exit':
            st.stop()

        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        if '$' in answer:
            answer = answer.replace('$', '\\$')
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write('Response: ')
        st.write(answer)
else:
    clear_directory()
    query = st.text_input("Questions: ")
    if query:

        st.session_state.messages.append({"role": "user", "content": query})

        if query == 'exit':
            clear_directory()
            st.stop()

        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        
        if '$' in answer: answer.replace('$', 'USD')

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write('Response: ')
        st.write(answer)
        query = ''
