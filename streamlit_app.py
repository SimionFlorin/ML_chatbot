import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from full_chain import create_full_chain, ask_question

datasource_list = [
    {
       "title": "Machine Learning Assistant",
       "collection_name": "machine_learning_course_doc",
       "subheader": "Talk with following course material:",
       "original_source_document":"Introduction-to-Machine-Learning.pdf",
       "original_source_document_link":"https://btu.edu.ge/wp-content/uploads/2023/04/Introduction-to-Machine-Learning-.docx.pdf"      
    }
]

index = 0
title = datasource_list[index]["title"]
collection_name = datasource_list[index]["collection_name"]
subheader = datasource_list[index]["subheader"]
original_source_document = datasource_list[index]["original_source_document"]
original_source_document_link = datasource_list[index]["original_source_document_link"]

st.set_page_config(page_title=title)
st.title(title)

def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user, "sources": []}]
    if "response_count" not in st.session_state:
        st.session_state.response_count = 0

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and i > 0:  # Skip the initial message
                response_index = st.session_state.response_count - (len(st.session_state.messages) - i - 1)
                show_sources = st.toggle(f"View sources for response {response_index}", key=f"toggle_{i}")
                if show_sources:
                    with st.container():
                        for j, source in enumerate(message.get("sources", [])):
                            expander = st.expander(f"View source {j+1}")
                            expander.markdown(f"**{source['metadata']['filename']}**")
                            expander.markdown(f"**{source['metadata']['part']}**")
                            expander.markdown(f"**{source['metadata']['chapter']}**")
                            sub_title = source['metadata'].get("sub_title", None)
                            if sub_title:
                                expander.markdown(sub_title)
                            expander.markdown(source['page_content'])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                print("prompt: ", prompt)
                print("qa: ", qa)
                response = ask_question(qa, prompt)
                st.markdown(response["answer"])
                
                sources = []
                for doc in response["context"]:
                    sources.append({
                        # "filename": doc.metadata.get('filename', ''),
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    })
                
                st.session_state.response_count += 1
                message = {"role": "assistant", "content": response["answer"], "sources": sources}
                st.session_state.messages.append(message)
                
                show_sources = st.toggle(f"View sources for response {st.session_state.response_count}", key=f"toggle_{len(st.session_state.messages)-1}")
                if show_sources:
                    with st.container():
                        for i, source in enumerate(sources):
                            expander = st.expander(f"View source {i+1}")
                            expander.markdown(f"**{source['metadata']['filename']}**")
                            expander.markdown(f"**{source['metadata']['part']}**")
                            expander.markdown(f"**{source['metadata']['chapter']}**")
                            sub_title = source['metadata'].get("sub_title", None)
                            if sub_title:
                                expander.markdown(sub_title)
                            expander.markdown(source['page_content'])

@st.cache_resource
def get_retriever(openai_api_key=None,qdrant_api_key=None, qdrant_url=None):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    embedding_model_name = "text-embedding-3-small"

    retriever = Qdrant(
        client=qdrant_client, 
        collection_name=collection_name,
        embeddings = OpenAIEmbeddings(model=embedding_model_name),
    ).as_retriever(search_kwargs={"k": 5})

    return retriever

def get_chain(openai_api_key=None, huggingfacehub_api_token=None,qdrant_api_key=None, qdrant_url=None):
    ensemble_retriever = get_retriever(openai_api_key=openai_api_key,qdrant_api_key=qdrant_api_key, qdrant_url=qdrant_url)
    chain = create_full_chain(ensemble_retriever,
                              openai_api_key=openai_api_key,
                              chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))
    return chain

def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        # st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        # if info_link:
        #     st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value

def run():
    ready = True

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")
    qdrant_url = st.session_state.get("QDRANT_URL")
    qdrant_api_key = st.session_state.get("QDRANT_API_KEY")

    # with st.sidebar:
    if not openai_api_key:
        openai_api_key = get_secret_or_input('OPENAI_API_KEY', "OpenAI API key",
                                                info_link="https://platform.openai.com/account/api-keys")
    if not huggingfacehub_api_token:
        huggingfacehub_api_token = get_secret_or_input('HUGGINGFACEHUB_API_TOKEN', "HuggingFace Hub API Token",
                                                        info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")
    if not qdrant_api_key:
        qdrant_api_key = get_secret_or_input('QDRANT_API_KEY', "Qdrant API Key",
                                                        info_link="https://qdrant.com/docs/quickstart/quickstart.html")
        
    if not qdrant_url:
        qdrant_url = get_secret_or_input('QDRANT_URL', "Qdrant URL",
                                                            info_link="https://qdrant.com/docs/quickstart/quickstart.html")


    if not openai_api_key:
        st.warning("Missing OPENAI_API_KEY")
        ready = False
    if not huggingfacehub_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        ready = False

    if ready:
        chain = get_chain(openai_api_key=openai_api_key, huggingfacehub_api_token=huggingfacehub_api_token, qdrant_api_key=qdrant_api_key, qdrant_url=qdrant_url)
        st.subheader(subheader)
        st.page_link(page=original_source_document_link,label= original_source_document)
        show_ui(chain, "What would you like to know?")
    else:
        st.stop()

run()