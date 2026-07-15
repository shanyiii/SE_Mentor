# SE Mentor

## Setup

### 1. 安裝依賴

```
docker compose up

uv sync

.venv\Scripts\activate
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