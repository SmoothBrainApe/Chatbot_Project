from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import LlamaCppEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from LLM_Functions import model_path, llm, template

text = "./data/Test.txt"
with open(text, "r") as file:
    contents = file.read()
embedding = LlamaCppEmbeddings(model_path=model_path, n_ctx=4096, verbose=True)

vectorstore = FAISS.from_texts([contents], embedding=embedding)
retriever = vectorstore.as_retriever()
template = template
prompt = ChatPromptTemplate.from_template(template)
model = llm

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

def RAG(message):
    response = chain.invoke(message)
    return response
