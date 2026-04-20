from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from pinecone import Pinecone
from dotenv import load_dotenv
from prompts import *
import uuid
import os

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

class PersonalChatBot:
    def __init__(self):
        self.store = {}
        self.session_id = str(uuid.uuid4())
        pc = Pinecone()
        self.index = pc.Index('anwaars-knowledge-base')
        self.model = self.create_model()
        self.embeddings = self.create_embeddings()
        self.retriever = self.create_vector_store()
        self.contextualize_prompt_templete, self.retrieval_prompt_templete = self.create_prompts()
        self.doc_chain = self.create_doc_chain()
        self.history_retriever = self.create_history_retriever()
        self.retrival_chain = self.crete_ret_chain()
        self.chain = self.create_final_chain()

    def create_model(self):
        hf_endpoints = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        return ChatHuggingFace(llm=hf_endpoints)

    def create_embeddings(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return embeddings
    
    def create_vector_store(self):
        vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 5})

    def create_prompts(self):
        contextualize_prompt_templete = ChatPromptTemplate.from_messages([
            ('system', contextualize_prompt),
            MessagesPlaceholder("chat_history"),
            ('human', '{input}')
        ])

        retrieval_prompt_templete = ChatPromptTemplate.from_messages([
            ('system', retrieval_prompt),
            MessagesPlaceholder("chat_history"),
            ('human', '{input}')
        ])
        return contextualize_prompt_templete, retrieval_prompt_templete
    
    def create_doc_chain(self):
        doc_chain = create_stuff_documents_chain(llm=self.model, prompt=self.retrieval_prompt_templete)
        return doc_chain

    def create_history_retriever(self):
        history_retriver = create_history_aware_retriever(llm=self.model, retriever=self.retriever, prompt=self.contextualize_prompt_templete)
        return history_retriver
    
    def my_session_history(self, session_id: str = "default") -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def crete_ret_chain(self):
        ret_chain = create_retrieval_chain(self.history_retriever, self.doc_chain)
        return ret_chain
    
    def create_final_chain(self):
        chain = RunnableWithMessageHistory(
            self.retrival_chain,
            self.my_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )
        return chain
    
    def invoke_chain(self, query):
        config = {"configurable": {"session_id": self.session_id}}
        response = self.chain.invoke({"input": query}, config=config)
        return response['answer']

# bot = PersonalChatBot()
# while True:
#     query = input("You: ")
#     if query == "exit":
#         break
#     response = bot.invoke_chain(query)
#     print("Bot: ", response)