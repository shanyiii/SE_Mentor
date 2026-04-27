import os, ast
from pydantic import BaseModel

from neo4j_impoter import Neo4jImoprter
from config import NEO4J_PASSWORD, OPENAI_API_KEY

from haystack.utils import Secret
from haystack import Document, Pipeline, component
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from neo4j_haystack import Neo4jEmbeddingRetriever, Neo4jDocumentStore

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class Question(BaseModel):
    question: str
    options: list[str]
    answer: int
    analysis: str
    concept: str

class QuestionList(BaseModel):
    questions: list[Question]

async def get_core_nodes():
    cypher = """
        MATCH (c:Entity)
        WITH c, count { (c)--() } AS degree
        ORDER BY degree DESC
        LIMIT 3
        RETURN c.name AS name
    """
    records = list()
    importer = Neo4jImoprter(uri="neo4j://localhost:7687", username="neo4j", password=NEO4J_PASSWORD)
    try:
        if importer.connect():
            records = importer.query_retrival(cypher)
            # print(records)
    finally:
        importer.close()
    return records

async def generate_quiz_kg() -> str:
    core_nodes = await get_core_nodes()
    if core_nodes:
        core_nodes_str = '、'.join(core_nodes)
        print(core_nodes_str)
    else:
        print("error: no nodes")
        return None

    document_store = Neo4jDocumentStore(
        url="neo4j://localhost:7687",
        username="neo4j",
        password=NEO4J_PASSWORD,
        database="neo4j",
        embedding_dim=384,
        index="document-embeddings",
    )

    template = [
        ChatMessage.from_user(
            """
            你是一個專業的「軟體工程」課程教授，請根據以下提供的教材內容，針對指定的核心概念設計三題單選題。

            【出題要求】：
            1. 題目必須具備鑑別度，測驗學生對該概念的理解而非單純記憶。
            2. 每一題有 4 個選項，並標註正確答案與詳細解析。
            3. 請替每一道題目備註對應的核心概念。

            Concept: {{concept}}

            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}
            """
        )
    ]

    prompt_builder = ChatPromptBuilder(template=template, required_variables=["concept"])
    gpt_chat = OpenAIChatGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"), 
        model="gpt-4o-mini",
        generation_kwargs={"response_format": QuestionList}
    )

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
            "text_embedder": {"text": core_nodes_str},
            "prompt_builder": {"concept": core_nodes_str}
        },
        include_outputs_from=["retriever", "llm"]
    )

    output_dicts = ast.literal_eval(result["llm"]["replies"][0]._content[0].text)
    q_dicts = [dict(q) for q in output_dicts["questions"]]
    return q_dicts

if __name__ == '__main__':    
    quizes = generate_quiz_kg()
    print(quizes)
    