import asyncio, json, os, statistics, re, sys
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple

import anthropic
from openai import AsyncOpenAI
from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy, AnswerAccuracy, AnswerCorrectness

sys.path.append(str(Path(__file__).resolve().parent.parent))
from haystack_controller import neo4j_retriever, neo4j_textbook_kg_retriever
from ragas_dataset import dataset
from config import CLAUDE_API_KEY, OPENAI_API_KEY

# os.environ["ANTHROPIC_API_KEY"] = CLAUDE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o", client=client)
# client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
# llm = llm_factory("claude-3-sonnet", client=client)

# Claude LLM
# claude_model = ChatAnthropic(
#     model="claude-3-5-haiku-latest",
#     temperature=0
# )
# llm = LangchainLLMWrapper(claude_model)

class LlmRetrieved(BaseModel):
    response: str
    retrieved_contents: list[str]

async def llm_retrieved_context(question: str, chapter: str) ->  Tuple[list[str], str]:
    file = await client.files.create(
        file=open(f"md_files\\textbooks\\ch{chapter}_markdown.md", "rb"),
        purpose="user_data"
    )
    prompt = f"""
    你是一個「軟體工程課程助教」請從提供的檔案裡尋找可以回答問題的資訊，並根據這些資訊回答問題，請用台灣繁體中文回答問題。
    請輸出你的回答，以及你檢索的資訊內容，可以檢索至多三個內容，檢索內容的總字數請限制在900字以內。
    請勿直接摘要資訊內容，而是針對問題僅採用必要資訊來回答，並控制在5到7句話，總字數不超過500字。
    如果問題與「軟體工程」無關，或是答案無法從資訊得知的話，請不要擅自生成答案。

    【問題】
    {question}
    """
    res = await client.responses.parse(
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
        text_format=LlmRetrieved
    )
    print(res.output_parsed)
    return res.output_parsed.retrieved_contents, res.output_parsed.response

async def get_retrieved_contexts(func, question: str, component_name: str, document_name: str, chapter: str = None) -> Tuple[list[str], str]:
    # result = func(question, chapter)
    result = func(question)
    retrieved_docs = result[component_name][document_name]
    # retrieved_contexts = [doc.content for doc in retrieved_docs]
    retrieved_contexts = [retrieved_docs]
    # response = result["llm"]["replies"][0]._content[0].text
    response = result["answer_llm"]["replies"][0]
    print("="*30)
    # print(response)
    return retrieved_contexts, response

async def context_precision(question: str, reference: str, retrieved_contexts: list[str]) -> float:
    if not retrieved_contexts:
        return 0.0

    # Create metric
    scorer = ContextPrecision(llm=llm)

    # Evaluate
    result = await scorer.ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=retrieved_contexts
    )
    print(f"- Context Precision Score: {result.value}")
    return result.value

async def context_recall(question: str, reference: str, retrieved_contexts: list[str]) -> float:
    if not retrieved_contexts:
        return 0.0
    
    # Create metric
    scorer = ContextRecall(llm=llm)

    # Evaluate
    result = await scorer.ascore(
        user_input=question,
        retrieved_contexts=retrieved_contexts,
        reference=reference
    )
    print(f"- Context Recall Score: {result.value}")
    return result.value

async def faithfulness(question: str, response: str, retrieved_contexts: list[str]) -> float:
    if not retrieved_contexts:
        return 0.0
    
    # Create metric
    scorer = Faithfulness(llm=llm)

    # Evaluate
    result = await scorer.ascore(
        user_input=question,
        response=response,
        retrieved_contexts=retrieved_contexts
    )
    print(f"- Faithfulness Score: {result.value}")
    return result.value

async def response_revelency(question: str, response: str) -> float:
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)
    scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)

    result = await scorer.ascore(
        user_input=question,
        response=response
    )
    print(f"- Answer Relevancy Score: {result.value}")
    return result.value

async def answer_accuracy(question: str, reference: str, response: str):
    scorer = AnswerAccuracy(llm=llm)

    # Evaluate
    result = await scorer.ascore(
        user_input=question,
        response=response,
        reference=reference
    )
    print(f"- Answer Accuracy Score: {result.value}")
    return result.value

async def answer_correctness(question: str, reference: str, response: str):
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)
    scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)

    # Evaluate
    result = await scorer.ascore(
        user_input=question,
        response=response,
        reference=reference
    )
    print(f"- Answer Correctness Score: {result.value}")
    return result.value

if __name__ == '__main__':
    # dataset = [
    # {
    #     "question": "如何進行專案預估？",
    #     "chapter":"5",
    #     "reference": "專案預估只需預估最底層需要執行的工作。先預估產品規模或複雜度，例如估計登入模組大約需要500行程式碼；再從規模或複雜度預估所需人力(工時)，例如寫500行程式約需15個小時；再來從需要的資源預估所需時程(工期)，例如15小時的工作約需要三天完成，因此開發登入模組這項工作可設定三天工期。"
    # },
    # {
    #     "question": "git add 和 git commit 差在哪裡？",
    #     "chapter":"6",
    #     "reference": "git add 是將檔案從工作目錄移到暫存區；git commit 是把暫存區的內容移至儲存庫。可以將暫存區比喻為購物車，git add 將商品放入購物車，而git commit 將購物車拿去結帳櫃檯。"
    # }]

    result_list = list()
    for i, data in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] 執行問題：{data['question']}")
        contexts, response = asyncio.run(get_retrieved_contexts(neo4j_textbook_kg_retriever, question=data["question"], component_name="desc_reasoner", document_name="knowledge_base"))
        # contexts, response = asyncio.run(get_retrieved_contexts(neo4j_retriever, question=data["question"], component_name="retriever", document_name="documents", chapter=data["chapter"]))
        # contexts, response = asyncio.run(llm_retrieved_context(data["question"], data["chapter"]))

        context_precision_score = asyncio.run(context_precision(data["question"], data["reference"], contexts))
        context_recall_score = asyncio.run(context_recall(data["question"], data["reference"], contexts))
        # faithfulness_score = asyncio.run(faithfulness(data["question"], response, contexts))
        # response_revelency_score = asyncio.run(response_revelency(data["question"], response))
        # answer_accuracy_score = asyncio.run(answer_accuracy(data["question"], data["reference"], response))
        answer_correctness_score = asyncio.run(answer_correctness(data["question"], data["reference"], response))
        average_score = statistics.fmean([context_precision_score, context_recall_score, answer_correctness_score])
        result_list.append({
                "question": data["question"],
                "reference_response": data["reference"],
                "llm_response": response,
                "retrieved_contexts": contexts,
                "context_precision_score": context_precision_score,
                "context_recall_score": context_recall_score,
                "answer_correctness_score": answer_correctness_score,
                "average_score": round(average_score, 2) 
        })

    current_dir = Path(__file__).parent
    output_file = current_dir.parent/"md_files"/"JSON"/"evaluation"/"rag_evaluation_kg_desc_ac.json"
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)

    # asyncio.run(llm_retrieved_context(dataset[0]["question"], dataset[0]["chapter"]))