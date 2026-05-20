import re

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

from langchain_opentutorial import set_env
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import OPENAI_API_KEY, LANGCHAIN_API_KEY
from common import FILEPATH

set_env(
    {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "LANGCHAIN_API_KEY": LANGCHAIN_API_KEY,
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "markdown-to-recursive",
    }
)

def pdf2md(file_path, chapter):
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(file_path)
    markdown_output = rendered.markdown
    output_name = f"md_files\\{chapter}_markdown.md"
    with open(output_name, "w", encoding="utf-8") as f:
        f.write(markdown_output)

def md_splitter(md_content):
    headers_to_split_on = [  
        ("#", "Header 1"),  
        ("##", "Header 2"),  
        ("###", "Header 3"),  
        ("####", "Header 4"),
    ]

    # Create a MarkdownHeaderTextSplitter object to split text based on markdown headers
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # Split markdown_document by headers and store in md_header_splits
    md_header_splits = markdown_splitter.split_text(md_content)

    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    final_chunks = recursive_splitter.split_documents(md_header_splits)

    # Print the split results
    # for doc in final_chunks:
    #     print(doc.page_content)
    #     print(doc.metadata)
    #     print("=" * 30)
    
    return final_chunks

def clean_markdown(text):
    # 移除圖片語法: ![替代文字](圖片連結)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # 移除連結語法，保留文字: [連結文字](連結) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # 移除上標/下標或特殊 HTML 標籤
    text = re.sub(r'<.*?>', '', text)
    # 移除過多的空格與換行
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

class Tags(BaseModel):
    tags: list[str] = Field(description="與文本有關的標籤列表")

def get_tags_from_gpt(course_name, level, doc_content):
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    parser = JsonOutputParser(pydantic_object=Tags)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位專業的{course_name}課程助教，請為以下內容提取3個{level}中文關鍵字標籤，並將3個標籤轉為英文一起輸出，如果原本的標籤是英文，則不須翻譯。"),
            ("user", "#Format: {format_instructions}\n\n#Content: {content}"),
        ]
    )

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser

    answer = chain.invoke({"course_name": course_name, "level": level, "content": doc_content})

    return answer

if __name__ == '__main__':
    file_names = ["[08]軟體設計-模組設計", "[09]軟體測試", "[10]進階軟體測試", "[11]DevOps自動化建置管理"]
    chapters = [8, 9, 10, 11]
    for file_name, ch in zip(file_names, chapters):
        pdf2md(f"C:\\Users\\shanyiii\\Desktop\\mine\\1141軟體工程\\slides\\{file_name}.pdf", f"ch{ch}")