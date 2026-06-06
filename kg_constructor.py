import re, ast, time, random, json, gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import anthropic

from file_processor import clean_markdown, md_splitter
from neo4j_impoter import Neo4jImoprter, TripleList
from common import NEO4J_URI
from config import NEO4J_PASSWORD, GEMINI_API_KEY, CLAUDE_API_KEY
from prompts import ENTITY_PROMPT_4_TEXTBOOK, TRIPLE_PROMPT_4_TEXTBOOK, ENTITY_PROMPT_4_SRD, TRIPLE_PROMPT_4_SRD
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

# client = OpenAI()
# client = genai.Client(api_key=GEMINI_API_KEY)
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# @retry(
#     stop=stop_after_attempt(10),            # 最多重試 5 次
#     wait=wait_exponential_jitter(initial=2, max=60), # 每次等待時間呈指數增加 (2秒, 4秒, 8秒...)
#     retry=retry_if_exception_type(ClientError), # 只有遇到 API ClientError 才重試
#     reraise=True
# )
def entities_extraction(user_input, prompt):
    # res = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role":"system", "content":prompt},
    #         {"role":"user", "content":user_input}
    #     ]
    # )

    res = client.beta.messages.create(
        max_tokens=1024,
        messages=[
            {"role":"assistant", "content":prompt},
            {"role":"user", "content":user_input}
        ],
        model="claude-sonnet-4-6",
    )

    # res = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     config=types.GenerateContentConfig(system_instruction=prompt),
    #     contents=user_input
    # )

    # entities = re.sub(r'[\r\n]', '', res.choices[0].message.content)  #gpt
    entities = re.sub(r'[\r\n]', '', res.content[0].text)   # claude
    # entities = re.sub(r'[\r\n]', '', res.text)    # gemini
    # print(entities)

    match = re.search(r"\[.*?\]", entities)
    if match:
        list_str = match.group()
        result = ast.literal_eval(list_str)  # 轉成真正的 list
        # print(result)
        return result
    else:
        print("找不到 list")
        return None

# @retry(
#     stop=stop_after_attempt(5),            # 最多重試 5 次
#     wait=wait_exponential_jitter(initial=2, max=60), # 每次等待時間呈指數增加 (2秒, 4秒, 8秒...)
#     retry=retry_if_exception_type(ClientError), # 只有遇到 API ClientError 才重試
#     reraise=True
# )
def relations_extraction(prompt, entities_list, user_input):
    doc_content = user_input.page_content if hasattr(user_input, 'page_content') else str(user_input)
    input_data = f"""
    實體列表如下：

    <entity>
    {entities_list}
    </entity>

    文章內容如下：

    <article>
    {doc_content}
    </article>
    """

    # res = client.responses.parse(
    #     model="gpt-4o",
    #     input=[
    #         {"role":"system", "content":prompt},
    #         {"role":"user", "content":input_data}
    #     ],
    #     text_format=TripleList
    # )

    res = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=4200,
        messages=[
            {"role":"user", "content":prompt},
            {"role":"user", "content":input_data}
        ],
        output_format=TripleList,
    )

    # res = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     config=types.GenerateContentConfig(
    #         system_instruction=prompt,
    #         response_mime_type="application/json",
    #         response_schema=TripleList,
    #     ),
    #     contents=input_data
    # )
    # print(res.output_parsed)
    # print(res.text)
    # return res.output_parsed  #gpt
    return res.parsed_output    # claude
    # return res.parsed     # gemini

if __name__ == '__main__':
    chapter_num = 11
    try:
        with open(f"md_files\\document\\海大餐飲外送系統-需求文件(SRD).md", 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")

    cleaned_md = clean_markdown(md_content)
    md_documents = md_splitter(cleaned_md)
    # print(input_data)

    doc_entity_pairs = list()
    page_entities_list = list()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_doc = {executor.submit(entities_extraction, doc.page_content, ENTITY_PROMPT_4_SRD): doc for doc in md_documents}
        
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                page_entities = future.result()
                if page_entities:
                    cleaned_page_entities = list(set(e.lower() for e in page_entities))
                    doc_entity_pairs.append((doc, cleaned_page_entities))
                    page_entities_list.extend(cleaned_page_entities)
            except Exception as e:
                print(f"實體抽取 Exception: {e}")

    with open(f"md_files\\JSON\\kgs\\doc\\doc_entities_g7_srd.json", 'w', encoding="utf-8") as f:
        json.dump(page_entities_list, f, indent=2, ensure_ascii=False)

    list_of_TripleList = list()
    print("實體抽取完成")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_relation = {
            executor.submit(relations_extraction, TRIPLE_PROMPT_4_SRD, entities, doc): doc 
            for doc, entities in doc_entity_pairs if entities
        }
        
        for future in as_completed(future_to_relation):
            try:
                triples = future.result()
                if triples:
                    list_of_TripleList.append(triples)
            except Exception as e:
                print(f"關係抽取 Exception: {e}")

    data_to_save = [t.model_dump(mode="json") for triples in list_of_TripleList for t in triples.triples]
    with open(f"md_files\\JSON\\kgs\\doc\\doc_triples_g7_srd.json", 'w', encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    
    print("關係抽取完成")
    triple_list = TripleList(triples=data_to_save)
    del list_of_TripleList
    gc.collect()

    # ======直接開啟抽取好的KG======

    # with open("md_files\\JSON\\kgs\\textbook_triples_ch9_shorter.json", 'r', encoding='utf-8') as f:
    #     triple_dict = json.load(f)
    # triple_list = TripleList(triples=triple_dict)
    
    # ======上傳KG======

    source_file = "海大餐飲外送系統-需求文件(SRD).md"
    
    importer = Neo4jImoprter(uri=NEO4J_URI, username="neo4j", password=NEO4J_PASSWORD)
    try:
        if importer.connect():
            is_success = importer.upload_doc_triples(triple_list, source_file, "SDD", "第七組")
            # is_success = importer.upload_textbook_triples(triple_list, source_file)
            print(f"上傳結果：{is_success}")
    finally:
        importer.close()