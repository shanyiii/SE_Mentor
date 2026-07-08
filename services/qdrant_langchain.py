from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import models
from google import genai
import requests, json

from file_processor import md_splitter, get_tags_from_gpt
from config import GEMINI_API_KEY, ML_HOST
from common import COLLECTION_NAME, QDRANT_URL, COURSE_NAME

embeddings = OpenAIEmbeddings()

# ollama
def get_tags_doc_level(doc_text):
    data = {
        "model": "llama3",
        "messages": [
            {'role': 'system', 'content': '你是一位專業的「軟體工程」課程助教，請為以下教材內容提取3個「檔案層級」的關鍵字標籤，以JSON格式輸出 {"tags": ["A", "B", "C"]}'},
            {'role': 'user', 'content': doc_text}
        ],
        "stream": False
    }
    resopnse = requests.post(ML_HOST, json=data)
    response_json = json.loads(resopnse.text)
    return response_json['message']['content']

def upload_to_qdrant(documents):

    vector_store = QdrantVectorStore.from_documents(
        documents=documents, 
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )
    print("===上傳完成===")

async def qdrant_retrieval(user_question):
    vector_store = QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    question_tags = get_tags_from_gpt(COURSE_NAME, "問題", user_question)
    print("[test] question tags: ", question_tags['tags'])
    
    print("===Retrieval start===")
    knowledge = ""
    results = vector_store.similarity_search(
            query=user_question,
            # filter=models.Filter(
            #     should=[
            #         models.FieldCondition(
            #             key="metadata.tags",
            #             match=models.MatchAny(
            #                 any=question_tags['tags']
            #             ),
            #         ),
            #     ]
            # )
        )
    for result in results:
        # knowledge = knowledge + f"{result.page_content}\n來源：{result.metadata}\n\n"
        knowledge = knowledge + f"{result.page_content}\n\n"
        print(result.page_content)
        # print("===tags===\n",result.metadata['tags'])
        print("="*30)
    print("===Retrieval completed===")
    
    query = f"資訊：\n{knowledge}\n\n你是一個「軟體工程課程助教」請將以上資訊作為基礎知識，回答問題，並僅輸出你的回答。如果問題與「軟體工程」無關，或是答案無法從以上資訊得知的話，請不要擅自生成答案。\n\n問題：{user_question}"
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            query
        ])

    return response.text
    # print(response.text)

if __name__ == '__main__':
    try:
        with open("md_files\\marker_test_output.md", 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")

    # doc_tags = get_tags_from_gpt(COURSE_NAME, "檔案層級", md_content)

    # doc_tags_json = get_tags_doc_level(md_content)  # tags at document level
    # doc_tags_dict = json.loads(doc_tags_json)
    # print("[debug] document level tags: ", doc_tags_dict)
    # print("="*30)

    md_documents = md_splitter(md_content)

    documents = list()
    print("===start chunk tags extraction===")
    for doc in md_documents:
        chunk_tags = get_tags_from_gpt(COURSE_NAME, "學術概念", doc.page_content)
        tags_list = [value for value in doc.metadata.values()]
        tags_list.extend(chunk_tags['tags'])
        content = f"文章標籤: {tags_list}\n\n{doc.page_content}"
        documents.append(Document(
            page_content=content
            # page_content=doc.page_content,
            # metadata={"tags": tags_list}
        ))
        # print(tags_list)

    print("===extraction finished===")

    upload_to_qdrant(documents)
    # qdrant_retrieval()