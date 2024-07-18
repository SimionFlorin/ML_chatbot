from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from memory import create_memory_chain


def create_full_chain(retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
    # model = get_model("ChatGPT", openai_api_key=openai_api_key)
    model = ChatOpenAI(model="gpt-4o-2024-05-13",temperature=0)
    
    initial_system_prompt = """You are a helpful AI assistant for busy professionals trying to improve their health.
    Use the following context and the users' chat history to help the user:
    If you don't know the answer, just say that you don't know. 
    
    Context: {context}
    
    Question: """

    system_prompt = """
    You are a machine learning engineer. You have some documents as a source of information. If no document answers the question, say you don't know.
    Respond to questions based on the following context and the conversation history:
    
    Context: {context}

    Question: """


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    chain = create_memory_chain(model, retriever, prompt, chat_memory)
    return chain


def ask_question(chain, query):
    print("query: ", query)
    response = chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": "foo"}}
    )
    print("response: ", response)
    return response

