import os
from typing import List, Iterable, Any

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from rag_chain import make_rag_chain



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_question(input):
    if not input:
        return None
    elif isinstance(input,str):
        return input
    elif isinstance(input,dict) and 'question' in input:
        return input['question']
    elif isinstance(input,BaseMessage):
        return input.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")

# def create_memory_chain(llm, base_chain, chat_memory):
def create_memory_chain(llm, retriever, rag_prompt, chat_memory):

### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """
    You are a machine learning engineer. You have some documents as a source of information. If no document answers the question, say you don't know.
    Answer questions based on the following context and the conversation history:
    
    Context: {context}

    Question:
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    ### Statefully manage chat history ###
    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

class SimpleTextRetriever(BaseRetriever):
    docs: List[Document]
    """Documents."""

    @classmethod
    def from_texts(
            cls,
            texts: Iterable[str],
            **kwargs: Any,
    ):
        docs = [Document(page_content=t) for t in texts]
        return cls(docs=docs, **kwargs)

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.docs
