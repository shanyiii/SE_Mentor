import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import anthropic

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    description: str
    
    def __hash__(self):
        return hash((self.subject, self.relation, self.object, self.description))
    
    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.subject == other.subject and 
                self.relation == other.relation and 
                self.object == other.object and
                self.description == other.description)
    
    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "description": self.description
        }
    
    def to_string(self) -> str:
        return f"({self.subject}, {self.relation}, {self.object}): {self.description}"


class KGEvaluator:
    """知識圖譜評估器"""
    
    def __init__(self, api_key: str = None):
        """初始化評估器"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-haiku-4-5"  # 使用較便宜的模型
        
    def load_triples_from_json(self, json_file: str) -> List[Triple]:
        """從 JSON 檔案載入三元組"""
        triples = []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支援多種 JSON 格式
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'triples' in data:
            items = data['triples']
        elif isinstance(data, dict) and 'data' in data:
            items = data['data']
        else:
            items = [data] if isinstance(data, dict) else []
        
        for item in items:
            if isinstance(item, dict):
                if 'subject' in item and 'relation' in item and 'object' in item:
                    subject_name = item['subject'].get('name', '') if isinstance(item['subject'], dict) else str(item['subject'])
                    relation_name = item['relation'].get('name', '') if isinstance(item['relation'], dict) else str(item['relation'])
                    object_name = item['object'].get('name', '') if isinstance(item['object'], dict) else str(item['object'])
                    subject_description = item['subject'].get('properties', '')[0].get('value', '') if  item['subject'].get('properties', '')[0]['key'] == "description" else ''
                    relation_description = item['relation'].get('description', '')
                    object_description = item['object'].get('properties', '')[0].get('value', '') if  item['subject'].get('properties', '')[0]['key'] == "description" else ''
                    desctiprion = '\n'.join([subject_description, relation_description, object_description])
                    
                    if subject_name and relation_name and object_name:
                        triple = Triple(
                            subject=subject_name,
                            relation=relation_name,
                            object=object_name,
                            description=desctiprion
                        )
                        triples.append(triple)
        
        return triples
    
    def sample_triples(self, json_file: str, sample_size: int = 50, seed: int = None) -> List[Triple]:
        """隨機採樣三元組"""
        if seed is not None:
            random.seed(seed)
        
        triples = self.load_triples_from_json(json_file)
        
        if len(triples) < sample_size:
            print(f"警告：檔案中只有 {len(triples)} 個三元組，採樣 {len(triples)} 個")
            return triples
        
        sampled = random.sample(triples, sample_size)
        return sampled
    
    def generate_ground_truth_prompt(self, triple: Triple) -> str:
        """生成 LLM 提示，要求驗證三元組的正確性"""
        prompt = f"""
        你是知識圖譜品質評估專家。請你根據提供的描述，驗證以下三元組是否正確：

        描述：
        {triple.description}

        三元組：
        - 主體（Subject）：{triple.subject}
        - 關係（Relation）：{triple.relation}
        - 客體（Object）：{triple.object}

        請根據提供的描述及你的知識評估這個三元組：
        1. **是否正確**：True 或 False
        2. **信心度**：0-100 之間的數字（100 表示完全確定）
        3. **說明**：簡短的說明為什麼你認為這個三元組是正確或不正確的

        請以 JSON 格式回應，格式如下：
        {{
            "is_correct": True/False,
            "confidence": 85,
            "explanation": "簡短說明"
        }}

        只回應 JSON，不需要其他文字。
        """
        
        return prompt
    
    def generate_ground_truth(self, triples: List[Triple], batch_mode: bool = False) -> Dict[str, Any]:
        """使用 Claude API 生成 ground truth"""
        results = []
        
        print(f"開始生成 {len(triples)} 個三元組的 ground truth...")
        
        for i, triple in enumerate(triples, 1):
            print(f"  [{i}/{len(triples)}] 驗證: {triple.to_string()}")
            
            prompt = self.generate_ground_truth_prompt(triple)
            
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                response_text = message.content[0].text
                
                # 嘗試從回應中提取 JSON
                try:
                    # 尋找 JSON 部分
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        gt_data = json.loads(json_match.group())
                    else:
                        gt_data = json.loads(response_text)
                    
                    result = {
                        "triple": triple.to_dict(),
                        "is_correct": gt_data.get("is_correct", False),
                        "confidence": gt_data.get("confidence", 50),
                        "explanation": gt_data.get("explanation", ""),
                        "raw_response": response_text
                    }
                except json.JSONDecodeError:
                    # 如果 JSON 解析失敗，手動判斷
                    is_correct = "correct" in response_text.lower() or "true" in response_text.lower()
                    print("slef detect is_correct: ", is_correct)
                    result = {
                        "triple": triple.to_dict(),
                        "is_correct": is_correct,
                        "confidence": 50,
                        "explanation": response_text[:200],
                        "raw_response": response_text
                    }
                
                results.append(result)
                
            except Exception as e:
                print(f"    錯誤: {str(e)}")
                result = {
                    "triple": triple.to_dict(),
                    "is_correct": None,
                    "confidence": 0,
                    "explanation": f"API 錯誤: {str(e)}",
                    "raw_response": ""
                }
                results.append(result)
        
        return {
            "total": len(triples),
            "results": results,
            "correct_count": sum(1 for r in results if r["is_correct"] is True),
            "incorrect_count": sum(1 for r in results if r["is_correct"] is False)
        }
    
    def calculate_metrics(self, extracted_triples: List[Triple], ground_truth_results: Dict[str, Any]) -> Dict[str, float]:
        """計算 Precision、Recall 和 F1 分數"""
        
        # 將 ground truth 結果轉換為集合
        correct_triples = set()
        for result in ground_truth_results["results"]:
            if result["is_correct"]:
                triple = Triple(
                    subject=result["triple"]["subject"],
                    relation=result["triple"]["relation"],
                    object=result["triple"]["object"],
                    description=result["triple"]["description"]
                )
                correct_triples.add(triple)
        
        # 轉換提取的三元組為集合
        extracted_set = set(extracted_triples)
        
        # 計算指標
        true_positives = len(extracted_set & correct_triples)
        false_positives = len(extracted_set - correct_triples)  # 實際為負，預測為正
        false_negatives = len(correct_triples - extracted_set)  # 實際為正，預測為負
        
        # Precision = TP / (TP + FP)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        # Recall = TP / (TP + FN)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_correct": len(correct_triples),
            "total_extracted": len(extracted_set)
        }
    
    def save_results(self, ground_truth_results: Dict[str, Any], metrics: Dict[str, float], output_file: str = "kg_evaluation_results.json"):
        """保存評估結果"""
        results = {
            "ground_truth": ground_truth_results,
            "metrics": metrics
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # print(f"\n結果已保存到 {output_file}")
    
    def print_results(self, ground_truth_results: Dict[str, Any], metrics: Dict[str, float]):
        print("\n" + "="*60)
        print("知識圖譜評估結果")
        print("="*60)
        
        print(f"\nGround Truth 統計：")
        print(f"  - 總三元組數：{ground_truth_results['total']}")
        print(f"  - 正確三元組：{ground_truth_results['correct_count']}")
        print(f"  - 不正確三元組：{ground_truth_results['incorrect_count']}")
        
        print(f"\n評估指標：")
        print(f"  - Precision（精準率）：{metrics['precision']:.2%}")
        print(f"  - Recall（召回率）：{metrics['recall']:.2%}")
        print(f"  - F1 分數：{metrics['f1']:.4f}")
        
        print(f"\n詳細統計：")
        print(f"  - 真正例（TP）：{metrics['true_positives']}")
        print(f"  - 假正例（FP）：{metrics['false_positives']}")
        print(f"  - 假負例（FN）：{metrics['false_negatives']}")
        
        print("\n個別三元組評估：")
        print("-" * 60)
        for i, result in enumerate(ground_truth_results["results"], 1):
            triple = result["triple"]
            status = "正確" if result["is_correct"] else "不正確" if result["is_correct"] is False else "未知"
            confidence = result.get("confidence", 0)
            
            print(f"\n{i}. {status} (信心度: {confidence})")
            print(f"   三元組：({triple['subject']}, {triple['relation']}, {triple['object']})")
            if result.get("explanation"):
                print(f"   說明：{result['explanation']}")


def main():
    chapter_num = 11
    sample_size = 10
    current_dir = Path(__file__).parent

    json_file = current_dir.parent/"md_files"/"JSON"/"kgs"/"textbook"/f"textbook_triples_ch{chapter_num}.json"
    output_file = current_dir.parent/"md_files"/"JSON"/"evaluation"/f"kg_evaluation_results_ch{chapter_num}.json"

    # 檢查檔案是否存在
    if not Path(json_file).exists():
        print(f"錯誤：找不到檔案 {json_file}")
        sys.exit(1)
    
    evaluator = KGEvaluator(api_key=config.CLAUDE_API_KEY)
    
    # 隨機採樣三元組
    print(f"從 {json_file} 中隨機採樣 {sample_size} 個三元組...")
    sampled_triples = evaluator.sample_triples(json_file, sample_size=sample_size, seed=45)
    print(f"成功採樣 {len(sampled_triples)} 個三元組\n")
    
    # 生成 ground truth
    ground_truth_results = evaluator.generate_ground_truth(sampled_triples)
    
    # 計算指標
    metrics = evaluator.calculate_metrics(sampled_triples, ground_truth_results)
    
    evaluator.print_results(ground_truth_results, metrics)
    evaluator.save_results(ground_truth_results, metrics, output_file)


if __name__ == "__main__":
    main()