#!/usr/bin/env python3
"""
Router GRPO Reward Plugin - 支持 JSON 格式的奖励函数

支持两种输出格式:
1. JSON 格式 (推荐):
   {"action": "S"}
   {"action": "R"}

2. 旧格式 (兼容):
   S
   R

评估逻辑:
- S 决策: 使用预计算的 s_path_correct
- R 决策: 使用预计算的 r_path_correct

路径效率惩罚:
- S: 0 (最高效)
- R: -0.4 (最昂贵)
"""

import json as json_lib
import os
import re
import sys
from datetime import datetime
from typing import List, Dict, Any
from swift.plugin import orms, ORM

# 添加项目路径（运行时自动检测）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, PROJECT_ROOT)

# 设置日志
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/router_grpo/reward_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"reward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")



def _log_reward_event(event: Dict[str, Any]):
    """记录 reward 计算事件到日志文件"""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json_lib.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[RouterReward] Failed to write log: {e}")


class RouterFinalRewardORM(ORM):
    """
    Router GRPO 奖励函数 - 基于公式: reward = correctness - cost - waste

    核心设计理念:
    1. S 决策：使用预计算的 s_path_correct
    2. R 决策：使用预计算的 r_path_correct
    3. 加大 S 和 R 的差距，让模型学会"S 够用就选 S"

    Reward 表（当前参数: WRONG_PENALTY=-1.5, COST_R=0.0, WASTE_R=0.4）:

    | s_correct | r_correct | S     | R    |
    |-----------|-----------|-------|------|
    | True      | True      | 1.0   | 0.6  |
    | False     | True      | -1.5  | 1.0  |
    | False     | False     | -1.5  | -1.5 |

    关键设计:
    - WRONG_PENALTY = -1.5：加强 S 错误惩罚
    - COST_R = 0.0：R 成本
    - WASTE_R = 0.4：summary够了还R的浪费惩罚
    - S错 vs R对 差距: 2.5

    Expected kwargs (from dataset):
    - s_path_correct: List[bool]
    - r_path_correct: List[bool]
    - query_id: List[str]
    """

    # ============================================================
    # REWARD 参数配置 - 可根据实验需求调整
    # ============================================================

    # === 基础正确性 ===
    CORRECT_REWARD = 1.0     # 答对的基础奖励
    WRONG_PENALTY = -1.5     # 答错的惩罚（必须为负）

    # === 动作成本 (Cost) - 选择该动作的固有成本 ===
    COST_S = 0.0             # S 最便宜
    COST_R = 0.0             # R 成本

    # === 浪费惩罚（s_correct=True 时触发）===
    WASTE_R = 0.4            # summary够了还R的浪费惩罚

    # === 其他惩罚 ===
    FORMAT_ERROR_PENALTY = -1.0    # 格式错误

    def __init__(self):
        pass

    def _extract_decision(self, completion: str) -> str:
        """从模型输出中提取决策

        支持两种格式:
        1. JSON: {"action": "S"} 或 {"action": "R"}
        2. 旧格式: S 或 R
        """
        text = completion.strip()

        # 移除 thinking 部分
        if "<think>" in text and "</think>" in text:
            match = re.search(r"</think>\s*(.+)", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # 尝试解析 JSON 格式
        if "{" in text:
            # 方法1: 直接解析
            try:
                obj = json_lib.loads(text)
                action = obj.get("action", "").upper().strip()

                if action == "S":
                    return "S"
                elif action == "R":
                    return "R"
            except json_lib.JSONDecodeError:
                pass

            # 方法2: 提取 JSON 对象
            start_idx = text.find("{")
            if start_idx >= 0:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(text)):
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break

                if brace_count == 0 and end_idx > start_idx:
                    try:
                        json_str = text[start_idx:end_idx + 1]
                        obj = json_lib.loads(json_str)
                        action = obj.get("action", "").upper().strip()

                        if action == "S":
                            return "S"
                        elif action == "R":
                            return "R"
                    except json_lib.JSONDecodeError:
                        pass

        # 旧格式解析
        text_clean = re.sub(r'<\|[^|]+\|>$', '', text).strip()  # 移除 <|im_end|> 等
        text_upper = text_clean.upper()

        if text_upper == "S":
            return "S"
        elif text_upper == "R":
            return "R"
        else:
            return "UNKNOWN"

    def __call__(
        self,
        completions: List[str],
        s_path_correct: List[bool] = None,
        r_path_correct: List[bool] = None,
        query_id: List[str] = None,
        **kwargs
    ) -> List[float]:
        """计算奖励"""
        rewards = []

        # 从 kwargs 中提取 messages（如果存在）
        messages_list = kwargs.get("messages", None)
        
        for i, completion in enumerate(completions):
            decision = self._extract_decision(completion)

            # 获取该样本的信息
            s_correct = s_path_correct[i] if s_path_correct and i < len(s_path_correct) else False
            r_correct = r_path_correct[i] if r_path_correct and i < len(r_path_correct) else False
            qid = query_id[i] if query_id and i < len(query_id) else ""
            
            # 提取原生 prompt（从 messages 中）
            original_prompt = ""
            if messages_list and i < len(messages_list):
                messages = messages_list[i]
                if isinstance(messages, list) and len(messages) > 0:
                    # messages 格式: [{"role": "user", "content": prompt}]
                    first_msg = messages[0]
                    if isinstance(first_msg, dict) and "content" in first_msg:
                        original_prompt = first_msg["content"]

            # === 奖励计算逻辑: reward = correctness - cost - waste ===
            reward = 0.0

            if decision == "S":
                # S 决策: s_correct=True → +1.0, else → -1.5
                if s_correct:
                    reward = self.CORRECT_REWARD
                else:
                    reward = self.WRONG_PENALTY

            elif decision == "R":
                # R 决策
                if r_correct:
                    if s_correct:
                        # r_correct=True 且 s_correct=True（浪费）：1.0 - COST_R - WASTE_R = 0.6
                        reward = self.CORRECT_REWARD - self.COST_R - self.WASTE_R
                    else:
                        # r_correct=True 且 s_correct=False：1.0 - COST_R = 1.0
                        reward = self.CORRECT_REWARD - self.COST_R
                else:
                    # r_correct=False：-1.5 - COST_R = -1.5
                    reward = self.WRONG_PENALTY - self.COST_R
            else:
                # 格式错误
                reward = self.FORMAT_ERROR_PENALTY
                log_entry = {
                    "type": "FORMAT_ERROR",
                    "query_id": qid,
                    "completion": completion[:200],  # 截断
                    "reward": reward,
                }
                # prompt 放在最后
                if original_prompt:
                    log_entry["prompt"] = original_prompt
                _log_reward_event(log_entry)

            # 记录每个样本的 reward（包含完整 completion）
            log_entry = {
                "type": "REWARD",
                "query_id": qid,
                "decision": decision,
                "reward": reward,
                "s_correct": s_correct,
                "r_correct": r_correct,
                "completion": completion,  # 记录完整输出
            }
            # prompt 放在最后
            if original_prompt:
                log_entry["prompt"] = original_prompt
            _log_reward_event(log_entry)

            rewards.append(reward)

        return rewards


class RouterFormatORM(ORM):
    """格式检查奖励：确保输出格式正确

    支持两种格式:
    1. JSON: {"action": "S"} 或 {"action": "R"}
    2. 旧格式: S 或 R
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            text = completion.strip()

            # 移除 thinking 部分
            if "<think>" in text and "</think>" in text:
                match = re.search(r"</think>\s*(.+)", text, re.DOTALL)
                if match:
                    text = match.group(1).strip()

            # 移除末尾的特殊标记 (如 <|im_end|>)
            text = re.sub(r'<\|[^|]+\|>$', '', text).strip()

            # 检查 JSON 格式
            if "{" in text:
                try:
                    # 尝试直接解析
                    obj = json_lib.loads(text)
                    action = obj.get("action", "").upper()
                    if action in ("S", "R"):
                        rewards.append(1.0)
                        continue
                except json_lib.JSONDecodeError:
                    pass

                # 尝试提取 JSON
                start_idx = text.find("{")
                end_idx = text.rfind("}")
                if start_idx >= 0 and end_idx > start_idx:
                    try:
                        json_str = text[start_idx:end_idx + 1]
                        obj = json_lib.loads(json_str)
                        action = obj.get("action", "").upper()
                        if action in ("S", "R"):
                            rewards.append(1.0)
                            continue
                    except json_lib.JSONDecodeError:
                        pass

            # 检查旧格式 (允许末尾有额外字符)
            text_upper = text.upper()
            if text_upper == "S" or text_upper == "R" or text_upper.startswith("S ") or text_upper.startswith("R "):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards


# 注册奖励函数
orms['router_final'] = RouterFinalRewardORM
orms['router_format'] = RouterFormatORM


if __name__ == "__main__":
    # 测试
    print("Testing RouterFinalRewardORM and RouterFormatORM...")
    print()

    test_completions = [
        # JSON 格式
        '{"action": "S"}',
        '{"action": "R"}',
        # 旧格式
        "S",
        "R",
        # Thinking 模式
        "<think>\nAnalysis...\n</think>\n\n{\"action\": \"S\"}",
        '<think>Thinking...</think>\n{"action": "R"}',
        # 错误格式
        "Invalid",
        "Maybe S or R",
    ]

    # ========== 1. 测试 Format ORM ==========
    format_orm = RouterFormatORM()
    format_rewards = format_orm(test_completions)

    print("=== Format Rewards ===")
    for i, (comp, reward) in enumerate(zip(test_completions, format_rewards)):
        comp_short = comp[:50].replace("\n", "\\n") + ("..." if len(comp) > 50 else "")
        print(f"[{i}] {reward:.1f} <- {comp_short}")

    print()

    # ========== 2. 测试 Decision Extraction ==========
    final_orm = RouterFinalRewardORM()
    print("=== Decision Extraction ===")
    for comp in test_completions:
        decision = final_orm._extract_decision(comp)
        comp_short = comp[:40].replace("\n", "\\n") + ("..." if len(comp) > 40 else "")
        print(f"{decision:8s} <- {comp_short}")

    print()

    # ========== 3. 测试 Final ORM (带完整参数) ==========
    print("=== Final ORM Test (with mock data) ===")

    # 模拟训练数据
    mock_completions = [
        '{"action": "S"}',           # S 决策，s_correct=True -> 正确
        '{"action": "S"}',           # S 决策，s_correct=False -> 错误
        '{"action": "R"}',           # R 决策，r_correct=True -> 正确
        '{"action": "R"}',           # R 决策，r_correct=False -> 错误
        "Invalid format",            # 格式错误
    ]

    mock_s_path_correct = [True, False, True, False, False]
    mock_r_path_correct = [True, True, True, False, False]
    mock_query_ids = [
        "mab_Accurate_Retrieval_0_qa1",
        "mab_Accurate_Retrieval_1_qa2",
        "mab_Accurate_Retrieval_2_qa3",
        "mab_Accurate_Retrieval_3_qa4",
        "mab_Accurate_Retrieval_4_qa5",
    ]

    print("Testing with mock data...")
    print()

    rewards = final_orm(
        completions=mock_completions,
        s_path_correct=mock_s_path_correct,
        r_path_correct=mock_r_path_correct,
        query_id=mock_query_ids,
    )

    print("Results:")
    print("-" * 80)
    for i, (comp, reward) in enumerate(zip(mock_completions, rewards)):
        decision = final_orm._extract_decision(comp)
        s_c = mock_s_path_correct[i]
        r_c = mock_r_path_correct[i]
        comp_short = comp[:35] + ("..." if len(comp) > 35 else "")

        # 预期逻辑
        if decision == "S":
            expected = f"s_correct={s_c} -> reward={1.0 if s_c else -1.5:.1f}"
        elif decision == "R":
            if r_c:
                if s_c:
                    expected = f"r_correct={r_c}, s_correct={s_c} -> reward={0.6:.1f}"
                else:
                    expected = f"r_correct={r_c}, s_correct={s_c} -> reward={1.0:.1f}"
            else:
                expected = f"r_correct={r_c} -> reward={-1.5:.1f}"
        else:
            expected = "format_error -> reward=-1.0"

        print(f"[{i}] {decision:8s} reward={reward:5.2f}  {expected}")
        print(f"    completion: {comp_short}")

    print("-" * 80)
    print()
    print("=== Test Complete ===")


