import re, ast, time, random, json, gc, sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import anthropic

from file_processor import clean_markdown, md_splitter, replace_tables_in_text, remove_specific_sections
from neo4j_importer import Neo4jImporter, TripleList, EntityList
from common import NEO4J_URI
from config import NEO4J_PASSWORD, GEMINI_API_KEY, CLAUDE_API_KEY
from prompts import ENTITY_PROMPT_4_TEXTBOOK, TRIPLE_PROMPT_4_TEXTBOOK, ENTITY_PROMPT_4_SRD, TRIPLE_PROMPT_4_SRD, ENTITY_PROMPT_4_SDD, TRIPLE_PROMPT_4_SDD, ENTITY_PROMPT_4_STD, KG_EXAMINATION_PROMPT, ENTITY_EXAMINATION_PROMPT, MATCH_FR_US_PROMPT

# client = OpenAI()
# client = genai.Client(api_key=GEMINI_API_KEY)
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def entities_extraction(prompt, user_input):
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
        max_tokens=9000,
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

    examined_triples = self_kg_examination(KG_EXAMINATION_PROMPT, user_input, res.parsed_output)
    return examined_triples

    # return res.output_parsed  #gpt
    # return res.parsed_output    # claude
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

    # examined_entities = self_entity_examination(ENTITY_EXAMINATION_PROMPT, user_input, res.parsed_output)

    return res.parsed_output    # claude
    # return examined_entities

def self_entity_examination(prompt, doc_content, entities):
    input_data = f"""
    文章內容如下：

    <article>
    {doc_content}
    </article>

    實體列表如下：

    <entities>
    {entities}
    </entities>
    """

    res = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=9000,
        messages=[
            {"role":"user", "content":prompt},
            {"role":"user", "content":input_data}
        ],
        output_format=EntityList,
    )

    return res.parsed_output    # claude

def self_kg_examination(prompt, doc_content, triples):
    input_data = f"""
    文章內容如下：

    <article>
    {doc_content}
    </article>

    三元組如下：

    <triples>
    {triples}
    </triples>
    """

    res = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=9000,
        messages=[
            {"role":"user", "content":prompt},
            {"role":"user", "content":input_data}
        ],
        output_format=TripleList,
    )

    return res.parsed_output    # claude

def match_fr_to_us_llm(ambiguous_case, prompt):
    input_data = f"""
    需求功能實體如下：

    <user_story>
    {ambiguous_case['fr']}
    </user_story>

    候選使用者故事實體列表如下：

    <candidates>
    {ambiguous_case['candidates']}
    </candidates>
    """

    res = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=5000,
        messages=[
            {"role":"user", "content":prompt},
            {"role":"user", "content":input_data}
        ],
        output_format=EntityList,
    )

    return res.parsed_output

def convert_props_to_dict(raw_list):
    entity_list = list()
    for raw in raw_list.entities:
        # item = dict()
        # item["name"] = raw.name
        item = raw.model_dump()
        item["label"] = str(raw.label.value)
        props = item.get("properties")
        item["properties_dict"] = {prop["key"]: prop["value"] for prop in props} if props else {}
        entity_list.append(item)
    return entity_list

def get_specific_entity_by_label(entity_list, label):
    prop_list = list()
    for entity in entity_list:
        if entity["label"] == label:
            prop_list.append(entity)
    return prop_list

def extract_actor(text, known_actors):
    for actor in known_actors:
        if actor in text:
            return actor
    return None

def add_req_reference_to_properties(entity_dict: dict, req_ids: list):
    if not entity_dict.get('properties'):
        entity_dict['properties'] = []
    
    # 移除舊的 req_reference（如果有的話）
    entity_dict['properties'] = [
        p for p in entity_dict['properties'] 
        if p.get('key') != 'req_reference'
    ]
    
    # 把每個 req_id 作為獨立的 KeyValPair 加入
    for req_id in req_ids:
        entity_dict['properties'].append({
            'key': 'req_reference',
            'value': req_id
        })
    
    return entity_dict

def match_us_to_fr_v2(raw_entity_list):
    entity_list = convert_props_to_dict(raw_entity_list)
    print(len(entity_list))

    us_list = get_specific_entity_by_label(entity_list, "UserStory")
    fr_list = get_specific_entity_by_label(entity_list, "Requirement")
    print("us list: ", len(us_list))
    print("fr list: ", len(fr_list))

    llm_matches = list()

    for fr in fr_list:
        llm_result = match_fr_to_us_llm({'fr': fr, 'candidates': us_list}, MATCH_FR_US_PROMPT)
        # print(llm_result)
        llm_matches.append(llm_result)
    
    match_list = list()
    if llm_matches:
        match_dict = list()
        for case in llm_matches:
            match_dict.extend(convert_props_to_dict(case))

        # print(match_dict)

        for match in match_dict:
            if match['properties_dict'].get('us_id'):
                us_obj = match

                if not match['properties_dict'].get('req_reference'):
                    continue

                if 'req_reference_list' not in us_obj:
                    us_obj['req_reference_list'] = []
                us_obj['req_reference_list'].append(match['properties_dict']['req_reference'])

                if 'req_reference' not in us_obj['properties_dict']:
                    us_obj['properties_dict']['req_reference'] = []

                elif isinstance(us_obj['properties_dict']['req_reference'], str):
                    us_obj['properties_dict']['req_reference'] = [us_obj['properties_dict']['req_reference']]

                us_obj['properties_dict']['req_reference'].append(match['properties_dict']['req_reference'])
                
                match_list.append({
                    'us': us_obj,
                    'us_req_ids': us_obj['req_reference_list']
                })

    return match_list, fr_list

def create_actor_relationships(raw_entity_list):
    entity_list = convert_props_to_dict(raw_entity_list)
    print(len(entity_list))

    known_actors = get_specific_entity_by_label(entity_list, "Actor")
    known_actors = set(actor["name"] for actor in known_actors)
    print("Actors: ", known_actors)

    fr_list = get_specific_entity_by_label(entity_list, "Requirement")
    print("fr list: ", len(fr_list))
    
    actor_relations = [] 
    
    for fr in fr_list:
        fr_actor = extract_actor(fr['properties_dict']['description'], known_actors)
        
        if fr_actor:
            # 處理 Actor
            actor_relations.append({
                "subject": {"name": fr_actor, "label": "Actor"},
                "relation": {"name": "操作", "description": f"{fr_actor}可執行「{fr['name']}」"},
                "object": {"name": fr['name'], "label": "Requirement"}
            })
    
    return actor_relations

def match_us_to_fr(raw_entity_list, model, similarity_threshold=0.5):
    entity_list = convert_props_to_dict(raw_entity_list)
    print(len(entity_list))

    known_actors = get_specific_entity_by_label(entity_list, "Actor")
    known_actors = set(actor["name"] for actor in known_actors)
    print("Actors: ", known_actors)

    us_list = get_specific_entity_by_label(entity_list, "UserStory")
    fr_list = get_specific_entity_by_label(entity_list, "Requirement")
    print("us list: ", len(us_list))
    print("fr list: ", len(fr_list))

    us_texts = [u['properties_dict']['description'] for u in us_list]
    fr_texts = [f['properties_dict']['description'] for f in fr_list]
    
    us_embeddings = model.encode(us_texts, normalize_embeddings=True)
    fr_embeddings = model.encode(fr_texts, normalize_embeddings=True)
    
    confident_matches = []
    actor_relations = [] 
    
    for i, fr in enumerate(fr_list):
        fr_actor = extract_actor(fr['properties_dict']['description'], known_actors)
        
        # 第一層過濾：角色要相符（如果抓不到角色，就不過濾，全部當候選）
        if fr_actor:
            # 處理 Actor
            actor_relations.append({
                "subject": {"name": fr_actor, "label": "Actor"},
                "relation": {"name": "操作", "description": f"{fr_actor}可執行「{fr['name']}」"},
                "object": {"name": fr['name'], "label": "Requirement"}
            })
            candidates_idx = [j for j, u in enumerate(us_list) if extract_actor(u['properties_dict']['description'], known_actors) == fr_actor]
        else:
            candidates_idx = list(range(len(us_list)))
        
        if not candidates_idx:
            candidates_idx = list(range(len(us_list)))
        
        sims = fr_embeddings[i] @ us_embeddings[candidates_idx].T
        top_indices = np.argsort(sims)[-3:][::-1]
        
        top_scores = [sims[idx] for idx in top_indices]
        
        # 判斷是否「夠清楚」：最高分要夠高，且要明顯領先第二名
        if top_scores[0] >= similarity_threshold and (len(top_scores) == 1 or top_scores[0] - top_scores[1] >= 0.05):
            # 清楚的案例，直接採用
            best_idx = candidates_idx[top_indices[0]]
            us_obj = us_list[best_idx]

            if 'req_reference_list' not in us_obj:
                    us_obj['req_reference_list'] = []
            us_obj['req_reference_list'].append(fr['properties_dict']['req_id'])

            if 'req_reference' not in us_obj['properties_dict']:
                us_obj['properties_dict']['req_reference'] = []
            us_obj['properties_dict']['req_reference'].append(fr['properties_dict']['req_id'])
            confident_matches.append({
                'fr': fr,
                'us': us_obj,
                'us_req_ids': us_obj['req_reference_list'],
                # 'us_id': us['properties_dict']['us_id'],
                # 'us_name': us['name'],
                # 'fr_id': fr_list[best_idx]['properties_dict']['req_id'],
                # 'fr_name': fr_list[best_idx]['name'],
                'similarity': float(top_scores[0])
            })
        # else:
        #     # 模糊案例：分數太低，或前幾名分數太接近，交給 LLM 判斷
        #     ambiguous_cases.append({
        #         'fr': fr,
        #         'candidates': [us_list[candidates_idx[idx]] for idx in top_indices]
        #     })

    # 找還沒有配對的
    matched_fr_ids = {m['fr']['properties_dict']['req_id'] for m in confident_matches}
    all_fr_ids = {fr['properties_dict']['req_id'] for fr in fr_list}
    unmatched_fr_ids = all_fr_ids - matched_fr_ids
    unmatched_fr_list = [f for f in fr_list if f['properties_dict']['req_id'] in unmatched_fr_ids]

    llm_matches = list()

    if unmatched_fr_list:
        print(f"已配對 {len(matched_fr_ids)} 個，仍有 {len(unmatched_fr_list)} 個 FR 沒有對應的 US，交給 LLM 用完整 US 清單判斷")
    
        for fr in unmatched_fr_list:
            llm_result = match_fr_to_us_llm({'fr': fr, 'candidates': us_list}, MATCH_FR_US_PROMPT)
            # print(llm_result)
            llm_matches.append(llm_result)
        # print(llm_matches)

        us_id_set = set()
        match_list = list()
        if llm_matches:
            for case in llm_matches:
                match_dict = convert_props_to_dict(case)
                for i, match in enumerate(match_dict):
                    us_id = match['properties_dict']['us_id']
                    if us_id not in us_id_set:
                        match_list.append(case.entities[i])
                        us_id_set.add(us_id)
    
    return confident_matches, match_list, fr_list, actor_relations

if __name__ == '__main__':
    source_file = "海大餐飲外送系統-測試文件(STD).md"
    # chapter_num = 11
    doc_type = "STD"
    # try:
    #     with open(f"md_files\\document\\{source_file}", 'r', encoding='utf-8') as input_file:
    #         md_content = input_file.read()
    # except FileNotFoundError:
    #     print("Error: The specified file was not found.")
    #     sys.exit(0)

    # cleaned_md = clean_markdown(md_content)
    # cleaned_md = remove_specific_sections(cleaned_md)
    # md_documents = md_splitter(cleaned_md)
    # # print(input_data)

    # document_contents = list()
    # for doc in md_documents:
    #     table_extracted_string = replace_tables_in_text(doc.page_content)
        # document_contents.append(table_extracted_string)

    # document_content = '\n+++\n'.join(document_contents)
    # with open(f"md_files\\document\\processed_std_doc_g7.md", 'w', encoding='utf-8') as f:
    #     f.write(document_content)

    # doc_entity_pairs = list()
    # page_entities_list = list()

    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     future_to_doc = {executor.submit(entities_extraction, ENTITY_PROMPT_4_SRD, content): content for content in document_contents}
        
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

    # with open(f"md_files\\JSON\\kgs\\doc\\doc_entities_g7_srd.json", 'w', encoding="utf-8") as f:
    #     json.dump(page_entities_list, f, indent=2, ensure_ascii=False)

    # print("實體抽取完成")
    # list_of_TripleList = list()
    
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     future_to_relation = {
    #         executor.submit(relations_extraction, TRIPLE_PROMPT_4_SRD, doc, entities): doc 
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
    # with open(f"md_files\\JSON\\kgs\\doc\\doc_triples_g7_srd.json", 'w', encoding="utf-8") as f:
    #     json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    
    # print("關係抽取完成")
    # triple_list = TripleList(triples=data_to_save)
    # del list_of_TripleList
    # gc.collect()

    # =====提取測試文件實體=====

    # list_of_EntityList = list()
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     future_to_doc = {executor.submit(std_entities_extraction, ENTITY_PROMPT_4_STD, content): content for content in document_contents}
        
    #     for future in as_completed(future_to_doc):
    #         doc = future_to_doc[future]
    #         try:
    #             entities = future.result()
    #             if entities:
    #                 list_of_EntityList.append(entities)
    #         except Exception as e:
    #             print(f"測試文件實體抽取 Exception: {e}")

    # for content in document_contents:
    #     std_entities_extraction(content, ENTITY_PROMPT_4_STD)

    # data_to_save = [t.model_dump(mode="json") for entities in list_of_EntityList for t in entities.entities]
    # with open(f"md_files\\JSON\\kgs\\doc\\doc_entities_g7_std.json", 'w', encoding="utf-8") as f:
    #     json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    # print("測試文件實體抽取完成")

    # ======直接開啟抽取好的KG======

    # with open("md_files\\JSON\\kgs\\doc\\doc_triples_g7_sdd.json", 'r', encoding='utf-8') as f:
    #     triple_dict = json.load(f)
    # triple_list = TripleList(triples=triple_dict)

    # ======直接開啟抽取好的 std entities======

    with open("md_files\\JSON\\kgs\\doc\\doc_entities_g7_std.json", 'r', encoding='utf-8') as f:
        entity_dict = json.load(f)
    entity_list = EntityList(entities=entity_dict)

    # ======連接US跟FR======

    # actor_relations = create_actor_relationships(entity_list)
    # triple_list = TripleList(triples=actor_relations)

    # model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    # confident_matches, llm_matches, fr_list, actor_relations = match_us_to_fr(entity_list, model, 0.8)

    # seen_us_ids = set()
    # confident_us_entities = list()
    
    # for match in confident_matches:
    #     us_dict = match['us'].copy()  # 重要：要 copy，不要直接改原物件
    #     us_id = us_dict['properties_dict']['us_id']
        
    #     if us_id not in seen_us_ids:
    #         # 把累積的 req_ids 正式加入 properties
    #         add_req_reference_to_properties(us_dict, match['us_req_ids'])
            
    #         entity = Entity(**us_dict)
    #         confident_us_entities.append(entity)
    #         seen_us_ids.add(us_id)

    # llm_matches, fr_list = match_us_to_fr_v2(entity_list)
    # seen_us_ids = set()
    # confident_us = list()
    # idx = dict()
    # i = 0
    
    # for match in llm_matches:
    #     us_dict = match['us'].copy()  # 重要：要 copy，不要直接改原物件
    #     us_id = us_dict['properties_dict']['us_id']
        
    #     if us_id not in seen_us_ids:
    #         # 把累積的 req_ids 正式加入 properties
    #         add_req_reference_to_properties(us_dict, match['us_req_ids'])
            
    #         confident_us.append(us_dict)
    #         idx[us_id] = i
    #         seen_us_ids.add(us_id)
    #         i += 1
    #     else:
    #         for req_id in match['us_req_ids']:
    #             confident_us[idx[us_id]]['properties'].append({
    #                 'key': 'req_reference',
    #                 'value': req_id
    #             })

    # for us in confident_us:
    #     existing_refs = set()
    #     deduplicated_props = []
        
    #     for prop in us['properties']:
    #         if prop['key'] == 'req_reference':
    #             if prop['value'] not in existing_refs:
    #                 deduplicated_props.append(prop)
    #                 existing_refs.add(prop['value'])
    #         else:
    #             deduplicated_props.append(prop)
        
    #     us['properties'] = deduplicated_props

    # confident_us_entities = [Entity(**item) for item in confident_us]
    # fr_list_entities = [Entity(**item) for item in fr_list]
    
    # print("向量配對的 US：")
    # for i, d in enumerate(confident_us_entities):
    #     print(f"[{i+1}/{len(confident_us_entities)}]:\n{d}")
    #     print("-"*30)

    
    # data_to_upload = list()

    # print("LLM 配對的 US 數量：", len(llm_matches))
    # for i, d in enumerate(llm_matches):
    #     print(f"[{i+1}/{len(llm_matches)}]:\n{d}")
    #     print("-"*30)

    # data_to_upload = confident_us_entities + llm_matches + fr_list_entities
    # data_to_upload = confident_us_entities + fr_list_entities

    # for i, d in enumerate(data_to_upload):
    #     print(f"[{i+1}/{len(data_to_upload)}]:\n{d}")
    #     print("-"*30)

    # entity_list = EntityList(entities=data_to_upload)

    # data_to_save = [data.model_dump() for data in data_to_upload]

    # with open(f"md_files\\JSON\\kgs\\doc\\doc_entities_pairs_g7_srd.json", 'w', encoding="utf-8") as f:
    #     json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    # with open(f"md_files\\JSON\\kgs\\doc\\doc_entities_pairs_g7_srd.json", 'r', encoding="utf-8") as f:
    #     entity_dict = json.load(f)
    # entity_list = EntityList(entities=entity_dict)

    # ======上傳KG======

    importer = Neo4jImporter(uri=NEO4J_URI, username="neo4j", password=NEO4J_PASSWORD)
    
    try:
        if importer.connect():
            # is_success = importer.upload_doc_triples(triple_list, source_file, doc_type, "第七組")
            is_success = importer.upload_entities(entity_list, source_file, doc_type, "第七組")
            connect_is_success = importer.link_references_to_requirements("TestCase", doc_type, "第七組", "驗證")
            # is_success = importer.upload_textbook_triples(triple_list, source_file)
            print(f"上傳結果：{is_success}")
            print(f"連接結果：{connect_is_success}")
    finally:
        importer.close()
