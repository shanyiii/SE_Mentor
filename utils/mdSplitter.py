from langchain_opentutorial import set_env
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from config import OPENAI_API_KEY, LANGCHAIN_API_KEY

set_env(
    {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "LANGCHAIN_API_KEY": LANGCHAIN_API_KEY,
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "markdown-to-recursive",
    }
)

def splitter():
    with open("marker_test_output.md", 'r', encoding='utf-8') as input_file:
        md_content = input_file.read()

    headers_to_split_on = [  
        ("#", "Header 1"),  
        ("##", "Header 2"),  
        ("###", "Header 3"), 
    ]

    # Create a MarkdownHeaderTextSplitter object to split text based on markdown headers
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # Split markdown_document by headers and store in md_header_splits
    md_header_splits = markdown_splitter.split_text(md_content)

    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,)
    final_chunks = recursive_splitter.split_documents(md_header_splits)

    # Print the split results
    # for doc in final_chunks:
    #     print(doc.page_content)
    #     print(doc.metadata)
    #     print("=" * 30)
    
    return final_chunks

# if __name__ == '__main__':
#     splitter()