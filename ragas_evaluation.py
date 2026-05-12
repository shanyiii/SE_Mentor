import asyncio, json

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

from haystack_controller import neo4j_retriever

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

async def get_retrieved_contexts(question: str) -> list[str]:
    result = await neo4j_retriever(question)
    retrieved_docs = result["retriever"]["documents"]
    retrieved_contexts = [doc.content for doc in retrieved_docs]
    # for context in retrieved_contexts:
    #     print(context[:200])
    response = result["llm"]["replies"][0]._content[0].text
    print("="*30)
    # print(response)
    return retrieved_contexts, response

async def context_precision(question: str, reference: str, retrieved_contexts: list[str]) -> float:
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

if __name__ == '__main__':
    dataset = [
        {
            "question": "請問白箱測試跟黑箱測試的差異是什麼?",
            "reference": "白箱測試跟黑箱測試是軟體測試的一種形式。白箱測是為了解軟體內部結構後進行的測試，能有效測試程式細節。黑箱測試則無需了解軟體內部結構，僅透過介面操作輸入資料，觀察輸出解果是否符合軟體需求功能。因此兩者的差異在白箱關心程式內部細節，黑箱則關心輸入輸出的結果。"
        },
        {
            "question": "請問git要做合併有哪些指令",
            "reference": "merge會將另一個分支的提交歷史合併到目前的分支。rebase則會將另一分支作為目前分支的參考基準，把目前分支的提交接到另一個分支的頂端，即讓提交記錄看起來更線性。因此git合併分支的指令有merge跟rebase。"
        },
        {
            "question": "什麼是凝聚力跟偶合力",
            "reference": "凝聚力跟耦合力是軟體工程中評估系統設計的兩大指標。凝聚力是指模組內部功能或資料的相關程度。凝聚力高，代表模組內部元素的相關度高，容易維護且適合再利用；耦合力是指模組之間的獨立程度。耦合力低，代表模組間的相關程度較低，一個模組的變更不容易影響其他模組，容易維護且適合再利用。通常希望達到高凝聚力，低耦合力。"
        },
        {
            "question": "請告訴我scrum的流程",
            "reference": "scrum透過固定周期的衝刺(sprint)進行迭代開發。第一步由product owner(PO)定義產品需求；第二步為衝刺規劃會議(sprint planning)，規劃這次迭代要完成的目標，並選擇product backlog item(PBI)；第三步開始衝刺，在衝刺期間內進行開發；第四步為每日會議(daily scrum)，每日進行進展檢視與討論，會議應少於15分鐘；第五步衝刺審查會議(sprint review)，在衝刺結束前由團隊與相關人士(stackholder)對這次的衝刺進行討論；第六步為衝刺回顧會議(sprint retrospective)，團隊反思流程，回顧這次衝刺遇到的問題。"
        },
        {
            "question": "模組設計裡的組合跟聚合差在哪裡",
            "reference": "模組設計中，組合(composition)跟聚合(aggregation)都是has a或part of的關係。組合是指若Y包含於X，則Y不可被其他物件包含，且Y跟X的生命週期一致，當X被刪除時Y也會被刪除；聚合是Y被包含於X時，Y還可以被其他物件包含，當X被刪除時Y仍然存在。所以組合是強關聯，而聚合是弱關聯。"
        }
    ]

    result_list = list()
    for data in dataset:
        contexts, response = asyncio.run(get_retrieved_contexts(data["question"]))
        context_precision_score = asyncio.run(context_precision(data["question"], data["reference"], contexts))
        context_recall_score = asyncio.run(context_recall(data["question"], data["reference"], contexts))
        faithfulness_score = asyncio.run(faithfulness(data["question"], response, contexts))
        result_list.append({
                "question": data["question"],
                "reference_response": data["reference"],
                "llm_response": response,
                "retrieved_contexts": contexts,
                "context_precision_score": context_precision_score,
                "context_recall_score": context_recall_score,
                "faithfulness_score": faithfulness_score 
        })

    with open("md_files\\JSON\\rag_evaluation.json", 'w', encoding="utf-8") as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)