import os, ast, asyncio, random
import inspect

from pydantic import BaseModel
from database.neo4j_importer import Neo4jImporter
from database.mongo_controller import DiagnosisQuiz, init_mongo, LearningProfile
from config import NEO4J_PASSWORD, OPENAI_API_KEY
from common import NEO4J_URI

from haystack.utils import Secret
from haystack import Pipeline
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

document_store = Neo4jDocumentStore(
    url="neo4j://localhost:7687",
    username="neo4j",
    password=NEO4J_PASSWORD,
    database="neo4j",
    embedding_dim=384,
    index="document-embeddings",
)

async def get_core_nodes(chapter: str):
    # 從指定章節的知識圖譜中取得關係數最多的前五個節點
    cypher = f"""
        MATCH (n:Concept)
        WHERE '{chapter}.pdf' IN n.source_files
        WITH n, count {{ (n)--() }} AS degree
        ORDER BY degree DESC
        LIMIT 5
        RETURN n.name AS name
    """
    records = list()
    importer = Neo4jImporter(uri=NEO4J_URI, username="neo4j", password=NEO4J_PASSWORD)
    try:
        if importer.connect():
            records = importer.run_cypher(cypher)
    except Exception as e:
        print(f"[quiz_generator]: Got an exception when querying: {e}")
        return None
    finally:
        importer.close()
    return records

async def generate_quiz_kg(chapter: str) -> str:
    # 生成測驗題目
    core_nodes = await get_core_nodes(chapter)
    if core_nodes:
        core_nodes_str = '、'.join(core_nodes)
        print(core_nodes_str)
    else:
        print("error: no nodes")
        return None

    template = [
        ChatMessage.from_user(
            """
            你是一個專業的「軟體工程」課程教授，請根據以下提供的教材內容，針對指定的核心概念設計五題單選題。

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
    pipeline.add_component("retriever", Neo4jEmbeddingRetriever(document_store=document_store, scale_score=False))
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", gpt_chat)

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    result = pipeline.run(
        data={
            "text_embedder": {"text": core_nodes_str},
            "prompt_builder": {"concept": core_nodes_str},
            "retriever": {
                "filters": {
                    "field": "source_file",
                    "operator": "==",
                    "value": f"{chapter}.pdf"
                }
            }
        },
        include_outputs_from=["retriever", "llm"]
    )

    output_dicts = ast.literal_eval(result["llm"]["replies"][0]._content[0].text)
    q_dicts = [dict(q) for q in output_dicts["questions"]]
    return q_dicts

async def upload_quiz_to_mongo():
    # 生成測驗題目並儲存於資料庫中
    quiz_client = await init_mongo("TABotAI_quiz")
    core_chapters = ["[10]進階軟體測試", "[01]軟體危機與軟體流程"]    # 要生成題目的章節

    diagnosis_quizes = list()

    for chapter in core_chapters:
        print(f"正在生成 {chapter} 的題目...")
        quizes = await generate_quiz_kg(chapter)
        # print(quizes)
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

async def get_quizes() -> list:
    quiz_client = await init_mongo("TABotAI_quiz")
    question_list = list()

    chapters = ["[01]軟體危機與軟體流程", "[02]基礎需求工程", "[03]使用者故事分析", "[04]敏捷開發方法", "[05]基礎專案管理與看板", "[06]版本控制", "[07]軟體設計-系統設計", "[08]軟體設計-模組設計", "[09]軟體測試", "[10]進階軟體測試", "[11]DevOps自動化建置管理"]
    chapter_numbers = random.sample(range(0, 11), 5)
    
    # 隨機取五個章節，每個章節隨機取一個題目
    for cn in chapter_numbers:
        quiz_list = await DiagnosisQuiz.find({"chapter": chapters[cn]}).to_list()
        quesion = random.choice(quiz_list)
        question_list.append(quesion)

    # quiz_list = await DiagnosisQuiz.find({"chapter": "[04]敏捷開發方法"}).to_list()
    # numbers = random.sample(range(0, len(quiz_list)), 5)
    return question_list

async def upsert_test(name):
    # 測試更新學習檔案用
    quiz_client = await init_mongo("TABotAI_quiz")

    profile = await LearningProfile.find_one(LearningProfile.student_name == name)
    if profile is None:
        await LearningProfile(student_name=name, student_id='12345678').insert()

if __name__ == '__main__':    
    asyncio.run(upload_quiz_to_mongo())
    # asyncio.run(get_quizes())

    # asyncio.run(upsert_test('shanyiii'))

    # print(inspect.getsource(Neo4jEmbeddingRetriever))
    
