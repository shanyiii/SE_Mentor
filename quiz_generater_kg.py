import os, ast, asyncio
from pydantic import BaseModel

from neo4j_impoter import Neo4jImoprter
from mongo_controller import DiagnosisQuiz, init_mongo
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

def get_core_nodes(chapter: str):
    cypher = f"""
        MATCH (n:Concept)
        WHERE '{chapter}.pdf' IN n.source_files
        WITH n, count {{ (n)--() }} AS degree
        ORDER BY degree DESC
        LIMIT 3
        RETURN n.name AS name
    """
    records = list()
    importer = Neo4jImoprter(uri="neo4j://localhost:7687", username="neo4j", password=NEO4J_PASSWORD)
    try:
        if importer.connect():
            records = importer.query_retrival(cypher)
            # print(records)
    except Exception as e:
        print(f"[quiz_generator]: Got an exception when querying: {e}")
        return None
    finally:
        importer.close()
    return records

def generate_quiz_kg(chapter: str) -> str:
    core_nodes = get_core_nodes(chapter)
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

async def upload_quiz_to_mongo():
    quiz_client = await init_mongo("TABotAI_quiz")
    core_chapters = ["[04]敏捷開發方法"]
    for chapter in core_chapters:
        quizes = generate_quiz_kg(chapter)
        # print(quizes)
        diagnosis_quizes = list()
        for quiz in quizes:
            diagnosis_quiz = DiagnosisQuiz(
                question=quiz["question"],
                options=quiz["options"],
                answer=quiz["answer"],
                analysis=quiz["analysis"],
                concept=quiz["concept"],
                chapter=chapter
            )
            diagnosis_quizes.append(diagnosis_quiz)
    await DiagnosisQuiz.insert_many(diagnosis_quizes)

async def get_quizes():
    quiz_client = await init_mongo("TABotAI_quiz")
    quiz_list = await DiagnosisQuiz.find({"chapter": "[04]敏捷開發方法"}).to_list()
    for quiz in quiz_list:
        print(quiz.question)

if __name__ == '__main__':    
    asyncio.run(upload_quiz_to_mongo())
    # asyncio.run(get_quizes())
    