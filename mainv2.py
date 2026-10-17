from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv    
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import PineconeHybridSearchRetriever
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# file_path = input("entre the file you want to chat \n")

api_key = os.getenv("api_key")

loader = UnstructuredPDFLoader(
    "database/iso27001.pdf"
)

docs= loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap = 100
)

print(type(docs))

text = [doc.page_content for doc in docs]

chunks = splitter.create_documents(text)

# print(len(chunks))
# print(chunks[0])



chat_history= [
     SystemMessage(content = " you are a full AI assitant"),
]

"""emmbedding"""
embedding = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)
"""vectore store"""
index_name = "hybrid-search-chatbot"
pc = Pinecone(api_key= api_key)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name = index_name,
        dimension=384,
        metric = 'dotproduct',
        spec = ServerlessSpec(
            cloud = 'aws', region = 'us-east-1'
        ),
    )

index = pc.Index(index_name)



"""bm25 coder"""
from pinecone_text.sparse import BM25Encoder

bm25encoder = BM25Encoder().default()

texts = [chunk.page_content for chunk in chunks]

bm25encoder.fit(texts)

bm25encoder.dump("bm25_values.json")
""" retriver"""
# retriver = vector_store.as_retriever( search_type = 'similarity',search_kwargs={"k": 3})
retriver = PineconeHybridSearchRetriever(
    embeddings=embedding, index= index, sparse_encoder=bm25encoder, top_k=5
)

retriver.add_texts(texts)

# def QUESTION ( query : str):
#     result = retriver.invoke(query)
def QUESTION(query: str):
    results = retriver.invoke(query)
    return "\n\n".join(doc.page_content for doc in results)


"""MODEL"""

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


template = PromptTemplate(
    template="""You are a query reformulation assistant for a PDF question-answering system.

Your task is to rewrite the user's latest question into a standalone question that can be understood without the previous conversation.

Use the conversation history to resolve:
- pronouns such as "it", "they", "this", "that"
- references such as "the previous year", "above", "below", "the second one"
- follow-up questions such as "why?", "how?", "what about it?"
- omitted subjects or entities

Rules:
1. Preserve the user's original intent.
2. Do not answer the question.
3. Do not add information that is not present in the conversation.
4. Return ONLY the rewritten question BUT If the question is already standalone, return it unchanged.

Conversation History:
{chat_history}

Latest User Question:
{question}

Standalone Question:""",
input_variables=['chat_history','question']
)


while True:
    user_input = input("entre your query: \n")
    if user_input == 'exit':
        break
    prompt = template.invoke({'chat_history' : chat_history, 'question' : user_input})
    new_query = (model.invoke(prompt)).content
    relevant_chunks= QUESTION(new_query)
    chat_history.append(HumanMessage(user_input))
    template2 = PromptTemplate(
        template="""
                    You are a helpful AI assistant that answers questions using information from the provided PDF context.

                    Your job is to answer the user's question accurately using the retrieved context.

                    Rules:

                    1. Use the provided PDF context as the primary source of factual information.
                    2. Use the conversation history to understand references and follow-up questions.
                    3. Do not invent facts that are not supported by the PDF context.
                    4. If the answer cannot be found in the provided PDF context, clearly say that the information is not available in the PDF.
                    5. You may use reasoning to connect information explicitly provided in the context.
                    6. Keep the answer clear and directly address the user's question.
                    7. If useful, mention the relevant page or section from the provided context.
                    8. Do not mention "retrieval", "chunks", "vector database", or internal system processes.
                    9. Do not repeat the entire conversation.
                    10. Answer naturally as part of an ongoing conversation.

                    Conversation History:
                    {chat_history}

                    Retrieved PDF Context:
                    {context}

                    Current User Question:
                    {question}

                    Answer:""",
                    input_variables=['question','context','chat_history']
    )
    prompt2 = template2.invoke(
        {
            'question' : new_query,
            'context' : relevant_chunks,
            'chat_history' : chat_history
        }
    )
    answer = model.invoke(prompt2)
    chat_history.append(AIMessage(answer.content))
    print(answer.content)


