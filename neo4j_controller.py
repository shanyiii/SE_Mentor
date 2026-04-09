import os, asyncio
from typing import List

from opencc import OpenCC
from haystack import Document, Pipeline, component
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from neo4j_haystack import Neo4jEmbeddingRetriever, Neo4jDocumentStore

from file_processor import md_splitter, clean_markdown
from config import OPENAI_API_KEY, NEO4J_PASSWORD

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

document_store = Neo4jDocumentStore(
    url="neo4j://localhost:7687",
    username="neo4j",
    password=NEO4J_PASSWORD,
    database="neo4j",
    embedding_dim=384,
    index="document-embeddings",
)

# 目前用的上傳方法
def upload_to_neo4j(documents: list[Document]):
    document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)

    document_store.write_documents(documents_with_embeddings.get("documents"))

    print(document_store.count_documents())

# 替 chunk 加上 metadata 並轉為 Document 物件
def add_metadata(md_documents: list[Document], source_file: str) -> list[Document]:
    documents = list()
    for doc in md_documents:
        documents.append(
            Document(
                content=clean_markdown(doc.page_content),
                meta={
                    "content_type": "textbook",
                    "source_file": source_file
                }
            )
        )
        # doc.meta["page"]
    return documents

@component
class MetadataEnricher:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        for doc in documents:
            doc.meta["kg_label"] = "DomainKG"
            doc.meta["source_type"] = "pdf"
        return {"documents": documents}
# 暫時不用
def upload_to_vector_db(file_path: list):
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=1))   
    pipeline.add_component("enricher", MetadataEnricher())
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter.documents", "cleaner.documents")
    pipeline.connect("cleaner.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "enricher.documents")
    pipeline.connect("enricher.documents", "embedder.documents")
    pipeline.connect("embedder.documents", "writer.documents")

    pipeline.run({"converter": {"sources": file_path}})

async def neo4j_retriever(question: str) -> str:
    template = [
        ChatMessage.from_user(
            """
            你是一個「軟體工程課程助教」請根據提供的資訊用台灣繁體中文回答問題，並僅輸出你的回答。如果問題與「軟體工程」無關，或是答案無法從資訊得知的話，請不要擅自生成答案。

            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            Question: {{question}}
            Answer:
            """
        )
    ]

    prompt_builder = ChatPromptBuilder(template=template, required_variables=["question"])
    gpt_chat = OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    pipeline.add_component("retriever", Neo4jEmbeddingRetriever(document_store=document_store))
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", gpt_chat)

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    result = pipeline.run(
        data={
            "text_embedder": {"text": question}, 
            "prompt_builder": {"question": question}
        },
        include_outputs_from=["retriever", "llm"]
    )

    return result
    # return result["llm"]["replies"][0]._content[0].text

async def main():
    res = await neo4j_retriever("請問有哪些git指令可以做分支合併？")
    print(res)
    retrieved_docs = res["retriever"]["documents"]
    for doc in retrieved_docs:
        print(doc.content[:200])
    print("="*30)
    print(res["llm"]["replies"][0]._content[0].text)

if __name__ == '__main__':
    try:
        with open("md_files\\marker_test_output.md", 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")

    md_documents = md_splitter(md_content)
    # documents = [Document(content=clean_markdown(doc.page_content)) for doc in md_documents]
    documents = add_metadata(md_documents, "[06]版本控制.pdf")

    upload_to_neo4j(documents)

    # upload_to_vector_db(["C:\\Users\\shanyiii\\Desktop\\mine\\1141軟體工程\\[06]版本控制.pdf"])

    # asyncio.run(main())