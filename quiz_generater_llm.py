import os
from openai import OpenAI
from pydantic import BaseModel
from config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

class Question(BaseModel):
    question: str
    options: list[str]
    answer: int
    analysis: str
    concept: str

class QuestionList(BaseModel):
    questions: list[Question]

async def generate_quiz_llm():
    file = client.files.create(
        file=open("md_files\\marker_test_output.md", "rb"),
        purpose="user_data"
    )

    prompt = f"""
    你是一個專業的「軟體工程」課程教授，請根據以下提供的教材內容，針對核心概念設計三題單選題。

    【出題要求】：
    1. 題目必須具備鑑別度，測驗學生對該概念的理解而非單純記憶。
    2. 每一題有 4 個選項，並標註正確答案與詳細解析。
    3. 請替每一道題目備註對應的核心概念。
    """

    res = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role":"system", "content":prompt},
            {"role":"user", "content":[
                    {
                        "type": "input_file",
                        "file_id": file.id,
                    }
                ]
            }
        ],
        text_format=QuestionList
    )
    q_dicts = [dict(q) for q in res.output_parsed.questions]
    return q_dicts

# if __name__ == '__main__':
#     outline = """
#     - 版本控管概念
#     - GIT基本指令
#     - GIT分支作法
#     - 使用遠端儲存庫GitHub
#     - GitHub Pull Request
#     - Git/GitHub開發流程
#     """
#     q_dicts = generate_quiz(outline)
#     print(q_dicts)