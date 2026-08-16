# SE Mentor

## Setup

### 1. 安裝依賴

```
docker compose up

uv sync

.venv\Scripts\activate
```

Docker 的 yaml 要修改 Noe4j 的密碼：

```
NEO4J_AUTH=neo4j/<密碼>
```

### 2. 設定 config.py

```
copy config_example.py config.py
```

編輯 `config.py`，填入以下設定：

```python!
# API keys
OPENAI_API_KEY = "<openai-api-key>"
GEMINI_API_KEY = "<gemini-api-key>"
CLAUDE_API_KEY = "<claude-api-key>"
LANGCHAIN_API_KEY = "<langchain-api-key>"

# dc 機器人的 token
DISCORD_TOKEN = "<discord-bot-token>"

# Neo4j 資料庫連線密碼
NEO4J_PASSWORD = "<your-neo4j-password>"
```

### 3. Mongodb

記得建立一個 Mongodb，collection 的名稱跟初始化資料庫的函式參數一樣。

```python!
# dc_chatbot.py

@bot.event
async def setup_hook():
    await init_mongo(<collection_name>)
```

## 使用說明

### dc_chatbot.py

跑機器人，直接執行即可。

```
uv run dc_chatbot.py
```

### services/kg_constructor.py

讀取檔案並建立知識圖譜。

#### 教材

```
uv run services/kg_constructor.py --textbook <file_path>
```

#### 開發文件

```
uv run services/kg_constructor.py <file_path> <doc_type> <group> <uploader>
```

- 'doc_type' 必須是 SRD、SDD、STD 其中一個

範例：

```
uv run services/kg_constructor.py files\test_file.md SRD 測試組 dev
```

### services/quiz_generator_kg.py

生成學習診斷測驗並儲存於資料庫，必須先生成測驗才能使用機器人的 `/quiz`。
根據需求修改程式碼 (如修改要生成測驗的章節)，直接執行就好了。

```
uv run services/quiz_generator_kg.py
```
