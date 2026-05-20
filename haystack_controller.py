import os, asyncio, json, re
from typing import List, Dict, Any

import anthropic
from opencc import OpenCC
from haystack import Document, Pipeline, component
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.anthropic import AnthropicGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from neo4j_haystack import Neo4jEmbeddingRetriever, Neo4jDocumentStore
from neo4j import GraphDatabase

from file_processor import md_splitter, clean_markdown
from config import OPENAI_API_KEY, NEO4J_PASSWORD, CLAUDE_API_KEY, GEMINI_API_KEY
from common import TASK_CONFIGS

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ----- custom components -----
@component
class SourceIdentifier:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    @component.output_types(filters=Dict[str, Any])
    def run(self, user_input: str):
        source_identify_prompt = f"""
        你是一個「軟體工程問題分類器」，請根據使用者的問題內容，判斷該問題可以從哪個軟體工程課程章節得到答案，並僅輸出章節名稱(包含編號)。

        【章節清單】
        - [01]軟體危機與軟體流程
        - [02]基礎需求工程
        - [03]使用者故事分析
        - [04]敏捷開發方法
        - [05]基礎專案管理與看板
        - [06]版本控制
        - [07]軟體設計-系統設計
        - [08]軟體設計-模組設計
        - [09]軟體測試
        - [10]進階軟體測試
        - [11]DevOps自動化建置管理

        【使用者問題】
        {user_input}
        """
        res = self.client.beta.messages.create(
            max_tokens=1024,
            messages=[
                {"role":"user", "content":source_identify_prompt}
            ],
            model="claude-opus-4-6",
        )
        print(f"來源資料標題：{res.content[0].text}")
        return {
            "filters": {
                "field": "source_file",
                "operator": "==",
                "value": f"{res.content[0].text}.pdf"
            }
        }

# 執行 Neo4j 查詢的自訂義 component
@component
class Neo4jExecutor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @component.output_types(query_result=str)
    def run(self, cypher_list: list[str]):
        # 移除 LLM 可能誤加的 markdown 語法
        cypher_query = " ".join(cypher_list)
        clean_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
        print(f"【final cypher】\n{clean_query}")
        
        with self.driver.session() as session:
            try:
                result = session.run(clean_query)
                # 取得結果並轉為易讀的字串格式
                records = [str(record.data()) for record in result]
                if not records:
                    return {"query_result": "在知識圖譜中找不到相關關聯。"}
                print(f"[debug] received records: \n{records}")
                return {"query_result": "\n".join(records)}
            except Exception as e:
                return {"query_result": f"Cypher 執行錯誤: {str(e)}"}

# ----- 初始化元件 -----
document_store = Neo4jDocumentStore(
    url="neo4j://localhost:7687",
    username="neo4j",
    password=NEO4J_PASSWORD,
    database="neo4j",
    embedding_dim=384,
    index="document-embeddings",
)
_embedder =  SentenceTransformersTextEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_claude = AnthropicGenerator(api_key=Secret.from_token(CLAUDE_API_KEY), model="claude-opus-4-6")
_gpt = OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")
_gpt_chat = OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")
_neo4j_retriever = Neo4jEmbeddingRetriever(document_store=document_store)
_neo4j_executor = Neo4jExecutor("bolt://localhost:7687", "neo4j", NEO4J_PASSWORD)

# ----- 上傳向量檔案 -----
# 目前用的上傳方法
def upload_to_neo4j(documents: list[Document]):
    document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)

    document_store.write_documents(documents_with_embeddings.get("documents"))

    print(document_store.count_documents())

# 替 chunk 加上 metadata 並轉為 Document 物件
def add_metadata(md_documents: list[Document], source_file: str, content_tags: list[str]) -> list[Document]:
    documents = list()
    for doc, content_tag in zip(md_documents, content_tags):
        cleaned_md = clean_markdown(doc.page_content)
        documents.append(
            Document(
                content=f"{content_tag}\n\n{cleaned_md}",
                meta={
                    "content_type": "textbook",
                    "source_file": source_file,
                    "tags": content_tag
                }
            )
        )
        # doc.meta["page"]
    return documents

# ----- haystack upload pipeline ----
# 暫時不用
@component
class MetadataEnricher:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        for doc in documents:
            doc.meta["kg_label"] = "DomainKG"
            doc.meta["source_type"] = "pdf"
        return {"documents": documents}

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

# ----- pipeline builders functions -----
def _build_retriever_pipeline(prompt_builder: ChatPromptBuilder) -> Pipeline:
    ranker = TransformersSimilarityRanker(top_k=3)

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", _embedder)
    pipeline.add_component("retriever", _neo4j_retriever)
    pipeline.add_component("ranker", ranker)
    pipeline.add_component("response_prompt", prompt_builder)
    # pipeline.add_component("source_identifier", SourceIdentifier(CLAUDE_API_KEY))
    pipeline.add_component("llm", _gpt_chat)
    # pipeline.add_component("llm", _claude)

    # pipeline.connect("source_identifier.filters", "retriever.filters")
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "ranker.documents")
    pipeline.connect("ranker.documents", "response_prompt.documents")
    # pipeline.connect("response_prompt.prompt", "llm.messages")
    pipeline.connect("response_prompt.prompt", "llm")

    return pipeline

def _build_kg_pipeline(cypher_prompt_builder: PromptBuilder) -> Pipeline:
    answer_prompt_template = """
    你是一位專業的軟體工程教學助教。請根據以下從知識圖譜檢索出來的「結構化資料」，回答學生的問題。

    【檢索資料】
    {{graph_results}}

    【學生問題】
    {{question}}

    如果有語義相同的資料，請合併為一個。例如：「會員清單瀏覽」跟「瀏覽會員清單」請合併為「瀏覽會員清單」。
    如果資料不足，請直接輸出「資料不足無法回答」。
    如果沒有資料，或是查詢語法有誤，請直接輸出「查無資料，請檢查查詢語法」。
    答案：
    """

    pipeline = Pipeline()
    pipeline.add_component("cypher_prompt", cypher_prompt_builder)
    pipeline.add_component("cypher_llm", _claude)
    pipeline.add_component("neo4j_executor", _neo4j_executor)
    pipeline.add_component("answer_prompt", PromptBuilder(template=answer_prompt_template))
    pipeline.add_component("answer_llm", _gpt)

    pipeline.connect("cypher_prompt", "cypher_llm")
    pipeline.connect("cypher_llm.replies", "neo4j_executor.cypher_list")
    pipeline.connect("neo4j_executor.query_result", "answer_prompt.graph_results")
    pipeline.connect("answer_prompt", "answer_llm")

    return pipeline

def neo4j_retriever(question: str, chapter: str) -> str:
    file_names = ["[01]軟體危機與軟體流程", "[02]基礎需求工程", "[03]使用者故事分析", "[04]敏捷開發方法", "[05]基礎專案管理與看板", "[06]版本控制", "[07]軟體設計-系統設計", "[08]軟體設計-模組設計", "[09]軟體測試", "[10]進階軟體測試", "[11]DevOps自動化建置管理"]
    chapter_name = file_names[int(chapter)-1]
    response_template = [
        ChatMessage.from_user(
            """
            你是一個「軟體工程課程助教」請根據提供的資訊用台灣繁體中文回答問題，並僅輸出你的回答。請勿直接摘要資訊內容，而是針對問題僅採用必要資訊來回答，並控制在5到7句話，總字數不超過500字。如果問題與「軟體工程」無關，或是答案無法從資訊得知的話，請不要擅自生成答案。

            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            Question: {{question}}
            Answer:
            """
        )
    ]

    pipeline = _build_retriever_pipeline(ChatPromptBuilder(template=response_template, required_variables=["question"]))

    result = pipeline.run(
        data={
            # "source_identifier": {"user_input": question},
            "text_embedder": {"text": question}, 
            "response_prompt": {"question": question},
            "ranker": {"query": question},
            "retriever": {
                "top_k": TASK_CONFIGS["retriever"]["top_k"],
                "filters": {
                    "field": "source_file",
                    "operator": "==",
                    "value": f"{chapter_name}.pdf"
                }
            },
            # "llm":{"generation_kwargs":{"max_tokens": TASK_CONFIGS["retriever"]["max_tokens"]}}
        },
        include_outputs_from=["retriever", "llm"]
    )

    return result
    # return result["llm"]["replies"][0]._content[0].text

def neo4j_generate_notes(concept: str) -> str:
    template = [
        ChatMessage.from_user(
            """
            請根據提供的「軟體工程」教材內容，針對指定的概念，生成一份筆記供學生學習，字數限制在1500字以內。
            筆記內容須包含：
            - 關鍵概念解說
            - 相關範例
            - 重點整理
            筆記內容必須清楚易懂，必要時可使用表格、引用等方式呈現。
            僅輸出筆記內容，並使用台灣繁體中文。

            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            Concept: {{concept}}
            Answer:
            """
        )
    ]

    pipeline = _build_retriever_pipeline(ChatPromptBuilder(template=template, required_variables=["concept"]))

    result = pipeline.run(
        data={
            "text_embedder": {"text": concept}, 
            "prompt_builder": {"concept": concept},
            "llm": {"generation_kwargs":{"max_tokens": TASK_CONFIGS["note"]["max_tokens"]}}
        },
        include_outputs_from=["retriever", "llm"]
    )

    return result

def neo4j_textbook_kg_retriever(question: str) -> str:
    # 將問題轉為 Cypher 的 Prompt
    cypher_prompt_template = """
    你是一位 Neo4j 專家。請根據以下 Schema 將使用者的問題轉化為精確的 Cypher 查詢語句，並以 string 形式輸出。

    【Schema 資訊】
    - Labels: Requirement, SystemComponent, API, TestCase, Actor, General, Concept, Technology, Methodology
    - Properties: name, group, source_files, description, (以及其他從文件提取的屬性)
    - Relationships: 是, 部分, 實作, 使用, 操作, 依賴, 改善, 解決

    【限制條件】
    - 只輸出 Cypher 語句，不要任何解釋或 markdown 標籤。
    - 盡量使用關鍵字模糊匹配 (CONTAINS) 以提高檢索成功率。

    使用者的問題: {{question}}
    生成的 Cypher:
    """

    pipeline = _build_kg_pipeline(PromptBuilder(template=cypher_prompt_template))

    result = pipeline.run(
        data={
            "cypher_prompt": {"question": question},
            "answer_prompt": {"question": question},
            "neo4j_executor": {}
        },
        include_outputs_from=["cypher_llm", "neo4j_executor", "answer_llm"]
    )
    print(f"【cypher from llm】:\n{result['cypher_llm']['replies']}")
    # return result
    return result["answer_llm"]["replies"][0]

def neo4j_doc_retriever(question: str, group_id: str) -> str:
    # 將問題轉為 Cypher 的 Prompt
    cypher_prompt_template = """
    你是一位 Neo4j 專家。請根據以下 Schema 將使用者的問題轉化為精確的 Cypher 查詢語句，並以 string 形式輸出。

    【Schema 資訊】
    - Labels: Requirement, SystemComponent, API, TestCase, Actor, General
    - Properties: name, group, source_files, description, (以及其他從文件提取的屬性)
    - Relationships: 是, 部分, 實作, 使用, 操作, 依賴, 改善, 解決

    【限制條件】
    - 必須包含 `group == '{{group}}'` 的過濾條件。
    - 只輸出 Cypher 語句，不要任何解釋或 markdown 標籤。
    - 盡量使用關鍵字模糊匹配 (CONTAINS) 以提高檢索成功率。

    使用者的問題: {{question}}
    生成的 Cypher:
    """

    pipeline = _build_kg_pipeline(PromptBuilder(template=cypher_prompt_template))

    result = pipeline.run(
        data={
            "cypher_prompt": {"question": question, "group": group_id},
            "answer_prompt": {"question": question},
            "neo4j_executor": {}
        },
        include_outputs_from=["cypher_llm", "neo4j_executor", "answer_llm"]
    )
    print(f"【cypher from llm】:\n{result['cypher_llm']['replies']}")
    # return result
    return result["answer_llm"]["replies"][0]

async def main():
    # res = await neo4j_retriever("請問有哪些git指令可以做分支合併？")
    res = await neo4j_generate_notes("Git Commits、Git Pull、Git Branches")
    # print(res)
    # retrieved_docs = res["retriever"]["documents"]
    # for doc in retrieved_docs:
    #     print(doc.content[:200])
    print("="*30)
    print(res["llm"]["replies"][0]._content[0].text)

async def batch_tags(md_documents):
    chunks = [d.page_content for d in md_documents]
    chunk_dict = json.dumps([{"id": i, "content": c} for i, c in enumerate(chunks)])

    prompt = f"""
    以下是 {len(md_documents)} 個軟體工程教材段落，請為每個段落生成 2-3 個關鍵字，並以 JSON 格式回傳，key 是段落編號，value 是關鍵字，以python list呈現。
    
    【輸出格式範例】
    1:[版本控制, git 指令]

    【段落】
    {chunk_dict}
    """

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    res = client.beta.messages.create(
            max_tokens=4096,
            messages=[
                {"role":"user", "content":prompt}
            ],
            model="claude-opus-4-6",
        )
    
    try:
        output = re.sub(r'```', '', res.content[0].text)
        output = re.sub(r'json', '', output)
        result = json.loads(output)
    except Exception as e:
        print(e)
        print("="*30)
        print(chunk_dict)
        print(output)
    # print(result)
    return result

async def upload_2_vectordb(chapter, textbook_name):
    try:
        with open(f"md_files\\textbooks\\ch{chapter}_markdown.md", 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        return

    md_documents = md_splitter(md_content)
    tags_w_id = await batch_tags(md_documents)
    chapter_tags = {
        "chapter": textbook_name,
        "tags": tags_w_id
    }
    with open(f"md_files\\JSON\\tags\\ch{chapter}_content_tags.json", 'w', encoding="utf-8") as f:
        json.dump(chapter_tags, f, indent=2, ensure_ascii=False)
    # with open("md_files\\JSON\\tags\\ch6_content_tags.json", 'r', encoding='utf-8') as f:
    #     tags_w_id = json.load(f)
    # documents = [Document(content=clean_markdown(doc.page_content)) for doc in md_documents] # 在 add_metadata 做
    content_tags = [tags_w_id[str(i)] for i in range(0, len(md_documents))]
    # print(content_tags[:5])
    documents = add_metadata(md_documents, f"{textbook_name}.pdf", content_tags)

    upload_to_neo4j(documents)

async def filter_retrieval_test():
    _embedder.warm_up()
    result = _embedder.run("什麼是凝聚力跟偶合力？")
    docs = _neo4j_retriever.run(
        query_embedding=result["embedding"],
        top_k=TASK_CONFIGS["filter_test"]["top_k"],
        filters={
            "field": "source_file",
            "operator": "==",
            "value": "[07]軟體設計-系統設計.pdf"
        }
    )
    for d in docs["documents"]:
        print(d.content)

if __name__ == '__main__':
    # file_names = ["[01]軟體危機與軟體流程", "[02]基礎需求工程", "[03]使用者故事分析", "[04]敏捷開發方法", "[05]基礎專案管理與看板", "[06]版本控制", "[07]軟體設計-系統設計", "[08]軟體設計-模組設計", "[09]軟體測試", "[10]進階軟體測試", "[11]DevOps自動化建置管理"]
    # chapters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # for file_name, ch in zip(file_names, chapters):
    #     asyncio.run(upload_2_vectordb(ch, file_name))

    # asyncio.run(filter_retrieval_test())

    question = "請問白箱測試跟黑箱測試的差異是什麼"
    res = neo4j_retriever(question, "9")
    for d in  res["retriever"]["documents"]:
        print(d.content)
    print("="*30)
    print(res["llm"]["replies"][0]._content[0].text)
    # print(res["llm"]["replies"][0])

    # upload_to_vector_db(["C:\\Users\\shanyiii\\Desktop\\mine\\1141軟體工程\\[06]版本控制.pdf"])

    # asyncio.run(main())