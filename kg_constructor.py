import re, ast
from openai import OpenAI

from file_processor import clean_markdown, md_splitter
from neo4j_impoter import Neo4jImoprter, TripleList
from config import NEO4J_PASSWORD

client = OpenAI()

def entities_extraction(user_input):
    prompt = """
    # Role
	你是一位資深的「軟體工程」專家，擅長將非結構化文本轉化為結構化的知識圖譜實體。

	# Task
	請從提供的文章中提取具有「教育價值」的知識點實體。這些實體必須具備與其他概念建立關聯的潛力。

	# Entity Categories (優先提取以下類別)
	- 架構與設計模式 (如: 微服務, 單例模式, 物件導向)
	- 技術與工具 (如: Docker, Kubernetes, Python, Git)
	- 開發流程與方法論 (如: CI/CD, Scrum, 測試驅動開發)
	- 核心概念與原理 (如: 耦合度, 內聚力)

	# Rules
	1. 去噪點：嚴禁提取「系統」、「文章」、「功能」、「用戶」、「免費」、「簡單」等過於一般化或描述性的詞彙。
	2. 標準化 (Normalization)：將同義詞映射至教材標準術語。例如：「版控」、「SVN/Git」統稱為「版本控制」；「寫 Code」統一為「程式編碼」。
	3. 具體性：實體必須是一個獨立的知識點。如果一個實體在脫離本文後無法定義為一個專業術語，請忽略它。
	4. 關係導向：只保留「可以被定義、被解釋、或與其他術語產生邏輯連接」的實體。
	5. 唯一性：確保輸出的 Python list 中沒有重複項。

	# Example
	- Input: "在進行敏捷開發時，我們常使用 Git 來進行版本管理，並透過 Jenkins 實作自動化部署。"
	- Output: ['敏捷開發', '版本控制', 'Git', 'Jenkins', '自動化部署']

	# Output Format
	請直接輸出一個 Python List，不需任何額外解釋。
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content":prompt},
            {"role":"user", "content":user_input}
        ]
    )

    entities = re.sub(r'[\r\n]', '', res.choices[0].message.content)
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

def relations_extraction(source_file, entities_list, user_input):
    prompt = f"""
    # Role
	你是一位知識工程師，專精於從軟體工程文本中建構語義網絡。

	# Task
	根據提供的「實體列表」，從「文章內容」中找出這些實體之間的邏輯關係，並以三元組 (Subject, Relation, Object) 的形式輸出。請連同來源 (source_file) 一同輸出。

	# Relation Schema (請優先使用以下關係語義)
	- 是：分類關係 (如: Git -> 是 -> 版本控制工具)
	- 部分：組成關係 (如: 記憶體管理 -> 部分 -> 作業系統)
	- 實作：實作/達成 (如: Scrum -> 實作 -> 敏捷開發)
	- 使用 / 依賴：技術依賴 (如: Web App -> 使用 -> HTTP 協定)
	- 改善 / 解決：解決特定問題 (如: 索引 -> 改善 -> 查詢效率)

	# Rules
	1. 嚴格限制：僅限於提取「實體列表」中出現的術語之間的關係。
	2. 方向性：確保 Subject 是主體，Object 是受體。例如：(Docker, implements, 容器化)，不可寫反。
	3. 精煉關係：關係詞應為動詞或動詞短語，且盡量使用上述 Schema 中的詞彙。
	4. 顯著性：只提取文章中明確支持的關係，避免主觀臆測。
	5. 連通性：若一個實體存在多個關係，請全部列出，以建立豐富的知識網路。
    """

    input_data = f"""
    來源檔案：{source_file}

    實體列表如下：

    <entity>
    {entities_list}
    </entity>

    文章內容如下：

    <article>
    {user_input}
    </article>
    """

    res = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role":"system", "content":prompt},
            {"role":"user", "content":input_data}
        ],
        text_format=TripleList
    )
    # print(res.output_parsed)
    print("關係抽取完成")
    return res.output_parsed

if __name__ == '__main__':
    try:
        with open("md_files\\shorter_markdown_test.md", 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")

    cleaned_md = clean_markdown(md_content)
    md_documents = md_splitter(cleaned_md)
    # # print(input_data)

    page_entities_list = list()
    for doc in md_documents:
        page_entities = entities_extraction(doc.page_content)
        if page_entities:
            page_entities_list.append(page_entities)
    
    entities_list = list(set(entity.lower() for page_entities in page_entities_list for entity in page_entities))
    print("實體抽取完成")
    # print(entities_list)

    triple_list = relations_extraction("[06]版本控制.pdf", entities_list, cleaned_md)
    
    importer = Neo4jImoprter(uri="neo4j://localhost:7687", username="neo4j", password=NEO4J_PASSWORD)
    try:
        if importer.connect():
            if_success = importer.upload_triples(triple_list)
            print(f"上傳結果：{if_success}")
    finally:
        importer.close()