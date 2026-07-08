import re, ast, time, random, json, gc, sys
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import anthropic

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import file_processor
from database.neo4j_importer import Neo4jImporter, TripleList, EntityList, Entity
from common import NEO4J_URI
from config import NEO4J_PASSWORD, GEMINI_API_KEY, CLAUDE_API_KEY
import prompts 

class KGConstructor:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    def entities_extraction(self, prompt: str, user_input: str) -> list:
        """LLM 提取實體 (SDD)"""

        # res = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[
        #         {"role":"system", "content":prompt},
        #         {"role":"user", "content":user_input}
        #     ]
        # )

        res = self.client.beta.messages.create(
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

    def relations_extraction(self, prompt: str, user_input: str, entities_list: list) -> TripleList:
        """LLM 提取三元組 (SDD)"""

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

        res = self.client.messages.parse(
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

        # examined_triples = self.self_kg_examination(prompts.KG_EXAMINATION_PROMPT, user_input, res.parsed_output)
        # return examined_triples

        # return res.output_parsed  #gpt
        return res.parsed_output    # claude
        # return res.parsed     # gemini

    def entities_extraction_with_properties(self, prompt: str, user_input: str) -> EntityList:
        """提取包含屬性的實體"""

        doc_content = user_input.page_content if hasattr(user_input, 'page_content') else str(user_input)
        input_data = f"""
        文章內容如下：

        <article>
        {doc_content}
        </article>
        """

        res = self.client.messages.parse(
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

    def self_entity_examination(self, prompt, doc_content, entities):
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

        res = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=9000,
            messages=[
                {"role":"user", "content":prompt},
                {"role":"user", "content":input_data}
            ],
            output_format=EntityList,
        )

        return res.parsed_output    # claude

    def self_kg_examination(self, prompt, doc_content, triples):
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

        res = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=9000,
            messages=[
                {"role":"user", "content":prompt},
                {"role":"user", "content":input_data}
            ],
            output_format=TripleList,
        )

        return res.parsed_output    # claude

    def match_fr_and_us_llm(self, case: dict, prompt: str, fr2us: bool) -> EntityList:
        """配對功能需求與使用者故事"""

        if fr2us:
            input_data = f"""
            需求功能實體如下：

            <user_story>
            {case['fr']}
            </user_story>

            候選使用者故事實體列表如下：

            <candidates>
            {case['candidates']}
            </candidates>
            """
        else:
            input_data = f"""
            使用者故事實體如下：

            <user_story>
            {case['us']}
            </user_story>

            候選需求功能實體列表如下：

            <candidates>
            {case['candidates']}
            </candidates>
            """

        res = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=5000,
            messages=[
                {"role":"user", "content":prompt},
                {"role":"user", "content":input_data}
            ],
            output_format=EntityList,
        )

        return res.parsed_output

    def convert_props_to_dict(self, raw_list: EntityList) -> list:
        """
        把類別屬性轉為字典格式
        """
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

    def get_specific_entity_by_label(self, entity_list: dict, label: str) -> list:
        """取得指定標籤的所有實體"""
        
        prop_list = list()
        for entity in entity_list:
            if entity["label"] == label:
                prop_list.append(entity)
        return prop_list

    def extract_actor(self, text: str, known_actors: list) -> str:
        """從描述中取得操作角色"""

        for actor in known_actors:
            if actor in text:
                return actor
        return None

    def add_req_reference_to_properties(self, entity_dict: dict, req_ids: list) -> dict:
        """
        更新實體裡的req_reference
        """

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

    def match_us_to_fr_v2(self, raw_entity_list: EntityList) -> tuple[list, list]:
        """LLM 去配對功能需求與使用者故事"""

        # 把 EntityList 轉為 dictionary
        entity_list = self.convert_props_to_dict(raw_entity_list)
        print(len(entity_list))

        # 取得使用者故事及需求
        us_list = self.get_specific_entity_by_label(entity_list, "UserStory")
        fr_list = self.get_specific_entity_by_label(entity_list, "Requirement")
        print("us list: ", len(us_list))
        print("fr list: ", len(fr_list))

        llm_matches = list()

        # 遍歷比較少的清單
        if len(fr_list) < len(us_list):
            for i, fr in enumerate(fr_list):
                print(f"[{i}/{len(fr_list)}] 配對中")
                # 遍歷功能需求讓 LLM 去配對對應的使用者故事
                llm_result = self.match_fr_and_us_llm({'fr': fr, 'candidates': us_list}, prompts.MATCH_FR_2_US_PROMPT, True)
                # print(llm_result)
                llm_matches.append(llm_result)
        else:
            for i, us in enumerate(us_list):
                print(f"[{i}/{len(us_list)}] 配對中")
                # 遍歷讓使用者故事 LLM 去配對對應的功能需求
                llm_result = self.match_fr_and_us_llm({'us': us, 'candidates': fr_list}, prompts.MATCH_US_2_FR_PROMPT, False)
                # print(llm_result)
                llm_matches.append(llm_result)
        
        match_list = list()
        if llm_matches:
            match_dict = list()
            for case in llm_matches:
                # 把 EntityList 轉為字典
                match_dict.extend(self.convert_props_to_dict(case))

            # print(match_dict)

            for match in match_dict:
                if match['properties_dict'].get('us_id'):   # 判斷是否是使用者故事
                    us_obj = match

                    if not match['properties_dict'].get('req_reference'):
                        continue

                    if 'req_reference_list' not in us_obj:
                        us_obj['req_reference_list'] = []
                    # 紀錄該使用者故事對應的功能需求編號
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

    def create_actor_relationships(self, raw_entity_list: list) -> list:
        """建立操作角色跟功能需求的三元組"""
        entity_list = self.convert_props_to_dict(raw_entity_list)

        # 取得所有操作角色 (標籤為 Actor)
        known_actors = self.get_specific_entity_by_label(entity_list, "Actor")
        known_actors = set(actor["name"] for actor in known_actors)
        print("Actors: ", known_actors)

        fr_list = self.get_specific_entity_by_label(entity_list, "Requirement")
        print("fr list: ", len(fr_list))
        
        actor_relations = list()
        
        for fr in fr_list:
            # 從功能需求中抓到操作角色
            fr_actor = self.extract_actor(fr['properties_dict']['description'], known_actors)
            
            if fr_actor:
                # 紀錄三元組「操作角色-操作->功能需求」
                actor_relations.append({
                    "subject": {"name": fr_actor, "label": "Actor"},
                    "relation": {"name": "操作", "description": f"{fr_actor}可執行「{fr['name']}」"},
                    "object": {"name": fr['name'], "label": "Requirement"}
                })
        
        return actor_relations

    def kg_construction_pipeline(self, document_contents: list, group: str) -> TripleList:
        """建立知識圖譜的流程"""
        doc_entity_pairs = list()
        page_entities_list = list()

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_doc = {executor.submit(self.entities_extraction, prompts.ENTITY_PROMPT_4_SDD, content): content for content in document_contents}
            
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
        
        with open(f"md_files\\groups\\{group}\\doc_entities_sdd.json", 'w', encoding="utf-8") as f:
            json.dump(page_entities_list, f, indent=2, ensure_ascii=False)

        print("實體抽取完成")
        list_of_TripleList = list()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_relation = {
                executor.submit(self.relations_extraction, prompts.TRIPLE_PROMPT_4_SDD, doc, entities): doc 
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
        with open(f"md_files\\groups\\{group}\\doc_triples_sdd.json", 'w', encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print("關係抽取完成")
        triple_list = TripleList(triples=data_to_save)
        del list_of_TripleList
        gc.collect()

        return triple_list

    def entities_extraction_pipeline(self, document_contents: list, doc_type: str, prompt: str, group: str):
        """文件實體抽取流程"""
        list_of_EntityList = list()
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_doc = {executor.submit(self.entities_extraction_with_properties, prompt, content): content for content in document_contents}
            
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    entities = future.result()
                    if entities:
                        list_of_EntityList.append(entities)
                except Exception as e:
                    print(f"文件實體抽取 Exception: {e}")

        data_to_save = [t.model_dump(mode="json") for entities in list_of_EntityList for t in entities.entities]
        with open(f"md_files\\groups\\{group}\\doc_entities_{doc_type.lower()}.json", 'w', encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        print("文件實體抽取完成")
        entity_list = EntityList(entities=data_to_save)

        return entity_list

    def match_fr_to_us_pipeline(self, entity_list: EntityList, group: str) -> EntityList:
        """配對功能需求跟使用者故事的流程"""
        llm_matches, fr_list = self.match_us_to_fr_v2(entity_list)
        seen_us_ids = set()
        confident_us = list()
        idx = dict()
        i = 0
        
        for match in llm_matches:
            us_dict = match['us'].copy()  # 重要：要 copy，不要直接改原物件
            us_id = us_dict['properties_dict']['us_id']
            
            if us_id not in seen_us_ids:
                # 把累積的 req_ids 正式加入 properties
                self.add_req_reference_to_properties(us_dict, match['us_req_ids'])
                
                confident_us.append(us_dict)
                idx[us_id] = i
                seen_us_ids.add(us_id)
                i += 1
            else:
                for req_id in match['us_req_ids']:
                    confident_us[idx[us_id]]['properties'].append({
                        'key': 'req_reference',
                        'value': req_id
                    })

        for us in confident_us:
            existing_refs = set()   # 移除重複的 req_reference
            deduplicated_props = []
            
            for prop in us['properties']:
                if prop['key'] == 'req_reference':
                    if prop['value'] not in existing_refs:
                        deduplicated_props.append(prop)
                        existing_refs.add(prop['value'])
                else:
                    deduplicated_props.append(prop)
            
            us['properties'] = deduplicated_props

        confident_us_entities = [Entity(**item) for item in confident_us]
        fr_list_entities = [Entity(**item) for item in fr_list]

        # print("向量配對的 US：")
        # for i, d in enumerate(confident_us_entities):
        #     print(f"[{i+1}/{len(confident_us_entities)}]:\n{d}")
        #     print("-"*30)
        
        data_to_upload = list()

        # print("LLM 配對的 US 數量：", len(llm_matches))
        # for i, d in enumerate(llm_matches):
        #     print(f"[{i+1}/{len(llm_matches)}]:\n{d}")
        #     print("-"*30)

        # data_to_upload = confident_us_entities + llm_matches + fr_list_entities   # 向量的
        data_to_upload = confident_us_entities + fr_list_entities   # LLM

        for i, d in enumerate(data_to_upload):
            print(f"[{i+1}/{len(data_to_upload)}]:\n{d}")
            print("-"*30)

        entity_list = EntityList(entities=data_to_upload)

        # 保存
        data_to_save = [data.model_dump() for data in data_to_upload]
        with open(f"md_files\\groups\\{group}\\doc_entities_pairs_srd.json", 'w', encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        return entity_list

if __name__ == '__main__':
    source_file = "海大教室借用平台-測試文件.md"
    chapter_num = 11
    doc_type = "STD"
    group = "第三組"

    try:
        with open(f"md_files\\document\\{source_file}", 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        sys.exit(0)
 
    cleaned_md = file_processor.clean_markdown(md_content)
    cleaned_md = file_processor.remove_specific_sections(cleaned_md)
    md_documents = file_processor.md_splitter(cleaned_md)

    document_contents = list()
    for doc in md_documents:
        table_extracted_string = file_processor.replace_tables_in_text(doc.page_content)
        document_contents.append(table_extracted_string)

    # document_content = '\n+++\n'.join(document_contents)
    # with open(f"md_files\\document\\processed_std_doc_g7.md", 'w', encoding='utf-8') as f:
    #     f.write(document_content)

    constructor = KGConstructor()
    importer = Neo4jImporter(uri=NEO4J_URI, username="neo4j", password=NEO4J_PASSWORD)
    try:
        if importer.connect():
            # =====建立知識圖譜 (實體+關係)=====
            if doc_type == "SDD":
                triple_list = constructor.kg_construction_pipeline(document_contents)
                is_success = importer.upload_doc_triples(triple_list, source_file, doc_type, group)
                is_success = importer.link_references_to_requirements("API", doc_type, group, "實作需求")

            # =====提取需求文件實體&配對=====
            elif doc_type == "SRD":
                # 抽實體
                entity_list = constructor.entities_extraction_pipeline(document_contents, doc_type, prompts.ENTITY_PROMPT_4_SRD)
                # 配對
                entity_list = constructor.match_fr_to_us_pipeline(entity_list)
                
                # ======直接開啟抽取好的 std entities======
                # with open("md_files\\JSON\\kgs\\doc\\doc_entities_g3_srd.json", 'r', encoding='utf-8') as f:
                #     entity_dict = json.load(f)
                # entity_list = EntityList(entities=entity_dict)

                is_success = importer.upload_entities(entity_list, source_file, doc_type, group)
                is_success = importer.link_references_to_requirements("UserStory", doc_type, group, "滿足")
                # 操作角色
                actor_relations = constructor.create_actor_relationships(entity_list)
                triple_list = TripleList(triples=actor_relations)
                is_success = importer.upload_doc_triples(triple_list, source_file, doc_type, group)
            
            # =====提取測試文件實體&連接需求文件=====    
            elif doc_type == "STD":
                # 抽實體
                entity_list = constructor.entities_extraction_pipeline(document_contents, doc_type, prompts.ENTITY_PROMPT_4_STD)
                is_success = importer.upload_entities(entity_list, source_file, doc_type, group)
                is_success = importer.link_references_to_requirements("TestCase", doc_type, group, "驗證")
                
            # is_success = importer.upload_textbook_triples(triple_list, source_file)
            print(f"上傳結果：{is_success}")
    except Exception as e:
        print(f"建立【{doc_type}】知識圖譜時遇到錯誤：{e}")
    finally:
        importer.close()


    # ======直接開啟抽取好的KG======

    # with open("md_files\\JSON\\kgs\\doc\\doc_triples_g7_sdd.json", 'r', encoding='utf-8') as f:
    #     triple_dict = json.load(f)
    # triple_list = TripleList(triples=triple_dict)



    # ======連接US跟FR======


    # ======上傳KG======
    

