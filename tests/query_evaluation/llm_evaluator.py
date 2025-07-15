import os
import json
import asyncio
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """LLM评估结果"""
    accuracy_score: float  # 准确性评分 (0-10)
    completeness_score: float  # 完整性评分 (0-10)
    relevance_score: float  # 相关性评分 (0-10)
    clarity_score: float  # 清晰度评分 (0-10)
    overall_score: float  # 总体评分 (0-10)
    reasoning: str  # 评估理由
    strengths: str  # 优点
    weaknesses: str  # 不足
    suggestions: str  # 改进建议

class LLMEvaluator:
    """使用大模型评估查询回答质量的评估器"""
    
    def __init__(self, config_path: str = None):
        """初始化LLM评估器"""
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "..", "config", "model_config.json")
        self.client = None
        self.evaluation_config = None
        self._load_config()
        self._init_client()
    
    def _load_config(self):
        """加载配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 设置评估器配置
            self.evaluation_config = {
                "provider": "openai",
                "model": "gpt-4o-mini",  # 使用GPT-4o-mini进行评估
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            # 如果配置中有评估器设置，使用它
            if "evaluator" in config.get("llm", {}).get("agents", {}):
                self.evaluation_config.update(config["llm"]["agents"]["evaluator"])
                
        except Exception as e:
            print(f"⚠️ 加载配置失败，使用默认配置: {e}")
            self.evaluation_config = {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 2000
            }
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("未找到OPENAI_API_KEY环境变量")
        
        self.client = AsyncOpenAI(api_key=api_key)
    
    def _create_evaluation_prompt(self, query: str, response: str, query_type: str, expected_criteria: list) -> str:
        """创建评估提示词"""
        criteria_text = "、".join(expected_criteria) if expected_criteria else "无特定标准"
        
        prompt = f"""你是一个专业的学术问答系统评估专家。请对以下查询和回答进行全面评估。

**用户查询**: {query}
**查询类型**: {query_type}
**期望评估标准**: {criteria_text}

**系统回答**: 
{response}

请从以下5个维度对回答进行评估，每个维度给出0-10分的评分：

1. **准确性 (Accuracy)**: 回答中的信息是否准确、事实是否正确
2. **完整性 (Completeness)**: 回答是否涵盖了查询的所有关键方面
3. **相关性 (Relevance)**: 回答是否直接相关并满足用户需求
4. **清晰度 (Clarity)**: 回答是否表达清晰、逻辑性强、易于理解
5. **总体质量 (Overall)**: 综合考虑以上因素的整体评价

请按照以下JSON格式返回评估结果：

{{
  "accuracy_score": 数字(0-10),
  "completeness_score": 数字(0-10),
  "relevance_score": 数字(0-10),
  "clarity_score": 数字(0-10),
  "overall_score": 数字(0-10),
  "reasoning": "详细的评估理由，说明各个评分的依据",
  "strengths": "回答的主要优点",
  "weaknesses": "回答的主要不足",
  "suggestions": "具体的改进建议"
}}

注意：
- 评分要客观公正，基于学术标准
- 如果回答包含错误信息，准确性评分应显著降低
- 如果回答没有直接回应查询，相关性评分应降低
- 如果回答过于简单或过于复杂，相应调整完整性和清晰度评分
- 总体评分应综合反映回答的整体质量"""

        return prompt
    
    async def evaluate_response(
        self, 
        query: str, 
        response: str, 
        query_type: str = "general",
        expected_criteria: list = None
    ) -> EvaluationResult:
        """使用LLM评估查询回答质量"""
        try:
            # 创建评估提示词
            prompt = self._create_evaluation_prompt(query, response, query_type, expected_criteria or [])
            
            # 调用LLM进行评估
            completion = await self.client.chat.completions.create(
                model=self.evaluation_config["model"],
                messages=[
                    {"role": "system", "content": "你是一个专业的学术问答系统评估专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.evaluation_config["temperature"],
                max_tokens=self.evaluation_config["max_tokens"]
            )
            
            # 解析评估结果
            result_text = completion.choices[0].message.content.strip()
            
            # 提取JSON部分
            if "```json" in result_text:
                json_start = result_text.find("```json") + 7
                json_end = result_text.find("```", json_start)
                json_text = result_text[json_start:json_end].strip()
            elif "{" in result_text and "}" in result_text:
                json_start = result_text.find("{")
                json_end = result_text.rfind("}") + 1
                json_text = result_text[json_start:json_end]
            else:
                raise ValueError("无法从LLM响应中提取JSON格式的评估结果")
            
            # 解析JSON
            evaluation_data = json.loads(json_text)
            
            # 创建评估结果对象
            return EvaluationResult(
                accuracy_score=float(evaluation_data.get("accuracy_score", 0)),
                completeness_score=float(evaluation_data.get("completeness_score", 0)),
                relevance_score=float(evaluation_data.get("relevance_score", 0)),
                clarity_score=float(evaluation_data.get("clarity_score", 0)),
                overall_score=float(evaluation_data.get("overall_score", 0)),
                reasoning=evaluation_data.get("reasoning", ""),
                strengths=evaluation_data.get("strengths", ""),
                weaknesses=evaluation_data.get("weaknesses", ""),
                suggestions=evaluation_data.get("suggestions", "")
            )
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            print(f"原始响应: {result_text}")
            # 返回默认低分评估结果
            return EvaluationResult(
                accuracy_score=3.0,
                completeness_score=3.0,
                relevance_score=3.0,
                clarity_score=3.0,
                overall_score=3.0,
                reasoning="评估过程中出现JSON解析错误",
                strengths="无法评估",
                weaknesses="评估系统错误",
                suggestions="请检查评估系统配置"
            )
        
        except Exception as e:
            print(f"⚠️ LLM评估失败: {e}")
            # 返回默认低分评估结果
            return EvaluationResult(
                accuracy_score=2.0,
                completeness_score=2.0,
                relevance_score=2.0,
                clarity_score=2.0,
                overall_score=2.0,
                reasoning=f"评估过程中出现错误: {str(e)}",
                strengths="无法评估",
                weaknesses="评估系统故障",
                suggestions="请检查网络连接和API配置"
            )
    
    async def batch_evaluate(self, test_results: list) -> list:
        """批量评估多个测试结果"""
        evaluated_results = []
        
        for i, result in enumerate(test_results):
            print(f"🤖 LLM评估进度: {i+1}/{len(test_results)}")
            
            if hasattr(result, 'query') and hasattr(result, 'response') and result.response:
                # 进行LLM评估
                llm_eval = await self.evaluate_response(
                    query=result.query,
                    response=result.response,
                    query_type=getattr(result, 'category', 'general'),
                    expected_criteria=getattr(result, 'evaluation_criteria', [])
                )
                
                # 将LLM评估结果添加到测试结果中
                result.llm_evaluation = llm_eval
                
                # 等待一下避免API限制
                await asyncio.sleep(0.5)
            
            evaluated_results.append(result)
        
        return evaluated_results

if __name__ == "__main__":
    # 测试评估器
    async def test_evaluator():
        evaluator = LLMEvaluator()
        
        test_query = "Michael Porter的竞争战略论文发表在哪一年？"
        test_response = "Michael Porter的经典论文《竞争战略》(Competitive Strategy)发表于1980年。这本书奠定了现代竞争分析的基础，提出了著名的五力模型。"
        
        result = await evaluator.evaluate_response(
            query=test_query,
            response=test_response,
            query_type="basic_information",
            expected_criteria=["准确的年份信息", "相关的补充信息"]
        )
        
        print("🎯 评估结果:")
        print(f"准确性: {result.accuracy_score}/10")
        print(f"完整性: {result.completeness_score}/10")
        print(f"相关性: {result.relevance_score}/10")
        print(f"清晰度: {result.clarity_score}/10")
        print(f"总体评分: {result.overall_score}/10")
        print(f"评估理由: {result.reasoning}")
    
    asyncio.run(test_evaluator()) 