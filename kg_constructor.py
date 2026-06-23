import re, ast, time, random, json, gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import anthropic

from file_processor import clean_markdown, md_splitter, replace_tables_in_text, remove_specific_sections
from neo4j_importer import Neo4jImporter, TripleList, EntityList
from common import NEO4J_URI
from config import NEO4J_PASSWORD, GEMINI_API_KEY, CLAUDE_API_KEY
from prompts import ENTITY_PROMPT_4_TEXTBOOK, TRIPLE_PROMPT_4_TEXTBOOK, ENTITY_PROMPT_4_SRD, TRIPLE_PROMPT_4_SRD, ENTITY_PROMPT_4_SDD, TRIPLE_PROMPT_4_SDD, ENTITY_PROMPT_4_STD
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

# client = OpenAI()
# client = genai.Client(api_key=GEMINI_API_KEY)
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def entities_extraction(user_input, prompt):
    # res = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role":"system", "content":prompt},
    #         {"role":"user", "content":user_input}
    #     ]
    # )

    res = client.beta.messages.create(
        max_tokens=2500,
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
        try:
            result = ast.literal_eval(list_str)  # 轉成真正的 list
            return result
        except Exception as e:
            print(list_str)
        # print(result)
    else:
        print("找不到 list")
        return None

def relations_extraction(prompt, user_input, entities_list):
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
        max_tokens=6000,
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

def std_entities_extraction(prompt, user_input):
    doc_content = user_input.page_content if hasattr(user_input, 'page_content') else str(user_input)
    input_data = f"""
    文章內容如下：

    <article>
    {doc_content}
    </article>
    """

    res = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=7000,
        messages=[
            {"role":"user", "content":prompt},
            {"role":"user", "content":input_data}
        ],
        output_format=EntityList,
    )

    return res.parsed_output    # claude

if __name__ == '__main__':
    source_file = "海大餐飲外送系統-測試文件(STD).md"
    # chapter_num = 11
    doc_type = "STD"
    # try:
    #     with open(f"md_files\\document\\{source_file}", 'r', encoding='utf-8') as input_file:
    #         md_content = input_file.read()
    # except FileNotFoundError:
    #     print("Error: The specified file was not found.")

    # cleaned_md = clean_markdown(md_content)
    # cleaned_md = remove_specific_sections(cleaned_md)
    # md_documents = md_splitter(cleaned_md)
    # # print(input_data)

    # document_contents = list()
    # for doc in md_documents:
    #     table_extracted_string = replace_tables_in_text(doc.page_content)
    #     document_contents.append(table_extracted_string)

    # doc_entity_pairs = list()
    # page_entities_list = list()

    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     future_to_doc = {executor.submit(entities_extraction, content, ENTITY_PROMPT_4_SDD): content for content in document_contents}
        
    #     for future in as_completed(future_to_doc):
    #         doc = future_to_doc[future]
    #         try:
    #             page_entities = future.result()
    #             if page_entities:
    #                 cleaned_page_entities = list(set(e.lower() for e in page_entities))
    #                 doc_entity_pairs.append((doc, cleaned_page_entities))
    #                 page_entities_list.extend(cleaned_page_entities)
    #         except Exception as e:
    #             print(f"實體抽取 Exception: {e}")

    # with open(f"md_files\\JSON\\kgs\\doc\\doc_entities_g7_sdd.json", 'w', encoding="utf-8") as f:
    #     json.dump(page_entities_list, f, indent=2, ensure_ascii=False)

    # print("實體抽取完成")
    # list_of_TripleList = list()
    
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     future_to_relation = {
    #         executor.submit(relations_extraction, TRIPLE_PROMPT_4_SDD, doc, entities): doc 
    #         for doc, entities in doc_entity_pairs if entities
    #     }
        
    #     for future in as_completed(future_to_relation):
    #         try:
    #             triples = future.result()
    #             if triples:
    #                 list_of_TripleList.append(triples)
    #         except Exception as e:
    #             print(f"關係抽取 Exception: {e}")

    # data_to_save = [t.model_dump(mode="json") for triples in list_of_TripleList for t in triples.triples]
    # with open(f"md_files\\JSON\\kgs\\doc\\doc_triples_g7_sdd.json", 'w', encoding="utf-8") as f:
    #     json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    
    # print("關係抽取完成")
    # triple_list = TripleList(triples=data_to_save)
    # del list_of_TripleList
    # gc.collect()

    # =====提取測試文件實體=====

    # list_of_EntityList = list()
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     future_to_doc = {executor.submit(std_entities_extraction, content, ENTITY_PROMPT_4_STD): content for content in document_contents}
        
    #     for future in as_completed(future_to_doc):
    #         doc = future_to_doc[future]
    #         try:
    #             entities = future.result()
    #             if entities:
    #                 list_of_EntityList.append(entities)
    #         except Exception as e:
    #             print(f"測試文件實體抽取 Exception: {e}")

    # data_to_save = [t.model_dump(mode="json") for entities in list_of_EntityList for t in entities.entities]
    # with open(f"md_files\\JSON\\kgs\\doc\\doc_entities_g7_std.json", 'w', encoding="utf-8") as f:
    #     json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    # print("測試文件實體抽取完成")

    # ======直接開啟抽取好的KG======

    # with open("md_files\\JSON\\kgs\\doc\\doc_triples_g7_sdd.json", 'r', encoding='utf-8') as f:
    #     triple_dict = json.load(f)
    # triple_list = TripleList(triples=triple_dict)

    # ======直接開啟抽取好的 std entities======

    # with open("md_files\\JSON\\kgs\\doc\\doc_entities_g7_std.json", 'r', encoding='utf-8') as f:
    #     entity_dict = json.load(f)
    # entity_list = EntityList(entities=entity_dict)
    
    # # ======上傳KG======

    importer = Neo4jImporter(uri=NEO4J_URI, username="neo4j", password=NEO4J_PASSWORD)
    
    # try:
    #     if importer.connect():
    #         # is_success = importer.upload_doc_triples(triple_list, source_file, doc_type, "第七組")
    #         is_success = importer.upload_std_entities(entity_list, source_file, doc_type, "第七組")
    #         # is_success = importer.upload_textbook_triples(triple_list, source_file)
    #         print(f"上傳結果：{is_success}")
    # finally:
    #     importer.close()

    # =====連接設計文件與需求文件=====

    try:
        if importer.connect():
            is_success = importer.link_references_to_requirements("TestCase", doc_type, "第七組", "驗證")
            print(f"連接結果：{is_success}")
    finally:
        importer.close()