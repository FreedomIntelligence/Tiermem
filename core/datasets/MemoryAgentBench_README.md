# MemoryAgentBench 数据加载器使用说明

## 📋 概述

MemoryAgentBench是一个评估LLM Agent记忆能力的benchmark，包含4个主要数据集：
- **Accurate_Retrieval** - 准确检索
- **Test_Time_Learning** - 测试时学习
- **Long_Range_Understanding** - 长距离理解
- **Conflict_Resolution** - 冲突解决

## 🔧 安装依赖

```bash
pip install datasets
```

## 📥 数据获取

MemoryAgentBench数据来自HuggingFace：`ai-hyz/MemoryAgentBench`

### 方式1：自动下载（推荐）
数据加载器会自动从HuggingFace下载数据，首次运行会需要一些时间。

### 方式2：手动下载到本地
```bash
# 使用huggingface-cli下载
huggingface-cli download ai-hyz/MemoryAgentBench --local-dir /share/home/qmzhu/AGMS/data/MemoryAgentBench_data

# 或者使用Python
from datasets import load_dataset
dataset = load_dataset("ai-hyz/MemoryAgentBench", "Accurate_Retrieval")
dataset.save_to_disk("/share/home/qmzhu/AGMS/data/MemoryAgentBench_data/Accurate_Retrieval")
```

## 🚀 使用方法

### 基本用法

```python
from core.datasets import memory_agent_bench

# 迭代sessions
for session in memory_agent_bench.iter_sessions(
    split="Accurate_Retrieval",
    sub_dataset="longmemeval_s_-1_500",  # 子数据集名称
    limit=10  # 限制样本数（用于测试）
):
    print(f"Session ID: {session['session_id']}")
    print(f"Number of turns: {len(session['turns'])}")
    print(f"Number of QA pairs: {len(session['qa_pairs'])}")
```

### 参数说明

- `split`: 主数据集名称
  - `"Accurate_Retrieval"`
  - `"Test_Time_Learning"`
  - `"Long_Range_Understanding"`
  - `"Conflict_Resolution"`

- `sub_dataset`: 子数据集名称（用于过滤source字段）
  - 如果不指定，默认使用split名称
  - 示例：`"longmemeval_s_-1_500"`, `"ruler_qa1_197k"`等

- `data_dir`: 本地数据集路径（如果已下载）
  - 默认：`/share/home/qmzhu/AGMS/data/MemoryAgentBench_data`

- `limit`: 限制返回的样本数量（用于测试）

### 返回格式

每个session包含：

```python
{
    "session_id": str,  # 唯一会话ID
    "turns": [  # 历史对话轮次（用于写入memory）
        {
            "speaker": "user",
            "text": str,  # context chunk
            "chunk_id": int
        }
    ],
    "qa_pairs": [  # 需要回答的问题
        {
            "query_id": str,
            "question": str,
            "ground_truth": str,
            "meta": {
                "sample_id": str,
                "qa_index": int,
                "qa_pair_id": str,
                "source": str
            }
        }
    ],
    "meta": {
        "num_context_chunks": int,
        "context_length": int,
        "num_qa_pairs": int,
        "source": str
    }
}
```

## 📊 可用的子数据集

### Accurate_Retrieval
- `longmemeval_s_-1_500`
- `longmemeval_s_star`
- `ruler_qa1_197k`
- `ruler_qa2_421k`
- `eventqa_64k`
- `eventqa_128k`
- `eventqa_full`

### Test_Time_Learning
- `icl_banking77`
- `icl_clinic150`
- `icl_nlu`
- `icl_trec_coarse`
- `icl_trec_fine`
- `recsys_redial_full`

### Long_Range_Understanding
- `detective_qa`
- `infbench_sum`

### Conflict_Resolution
- `factconsolidation_mh_6k`
- `factconsolidation_mh_32k`
- `factconsolidation_mh_64k`
- `factconsolidation_mh_262k`
- `factconsolidation_sh_6k`
- `factconsolidation_sh_32k`
- `factconsolidation_sh_64k`
- `factconsolidation_sh_262k`
- `conflict_resolution_custom`

## 🔍 测试数据加载器

```bash
cd /share/home/qmzhu/AGMS
python -m core.datasets.memory_agent_bench
```

## ⚠️ 注意事项

1. **首次运行较慢**：如果数据未下载，首次运行需要从HuggingFace下载，可能需要几分钟到几十分钟。

2. **网络要求**：需要能够访问HuggingFace。如果网络受限，建议先手动下载到本地。

3. **内存占用**：某些大型数据集（如`ruler_qa2_421k`）可能占用较多内存。

4. **子数据集名称**：必须使用正确的sub_dataset名称，否则可能找不到数据。可以参考官方配置文件：
   `/share/home/qmzhu/benchmark_github/MemoryAgentBench/configs/data_conf/`

## 🐛 问题排查

### 问题1：找不到数据集
```
Error: Split Accurate_Retrieval not found
```
**解决方案**：
- 检查网络连接
- 确认HuggingFace账号已登录（如果需要）
- 尝试手动下载到本地

### 问题2：sub_dataset不匹配
```
Filtered to 0 samples matching source 'xxx'
```
**解决方案**：
- 检查sub_dataset名称是否正确
- 查看官方配置文件确认可用的sub_dataset名称

### 问题3：内存不足
**解决方案**：
- 使用`limit`参数限制样本数
- 选择较小的子数据集
- 增加系统内存

## 📚 参考文档

- 官方README: `/share/home/qmzhu/benchmark_github/MemoryAgentBench/README.md`
- 数据加载代码: `/share/home/qmzhu/benchmark_github/MemoryAgentBench/utils/eval_data_utils.py`
- HuggingFace数据集: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench


