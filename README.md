# AGMS - 统一评估框架

## 📁 项目结构

```
AGMS/
├── core/
│   ├── datasets/          # 统一数据加载器
│   │   ├── locomo.py
│   │   ├── hotpotqa.py
│   │   └── memory_agent_bench.py
│   ├── systems/           # Baseline系统适配器
│   │   ├── base.py        # MemorySystem抽象接口
│   │   ├── rawllm_adapter.py
│   │   ├── mem0_adapter.py
│   │   ├── gam_adapter.py
│   │   ├── lightmem_adapter.py
│   │   ├── mirix_adapter.py
│   │   └── agms_system.py
│   └── runner/            # 评估框架
│       ├── run_benchmark.py
│       ├── logging_utils.py
│       └── scoring.py
├── data/                  # 数据集
├── relatedwork/           # Baseline框架
├── results/               # 实验结果（自动生成）
├── experiments.md         # 实验指导文档
└── 实验复现计划.md        # 详细实验计划
```

## 🚀 快速开始

### 1. 测试数据加载器

```bash
cd /share/home/qmzhu/AGMS

# 测试LoCoMo数据加载
python -m core.datasets.locomo

# 测试HotpotQA数据加载
python -m core.datasets.hotpotqa
```

### 2. 运行第一个baseline（RawLLM）

```bash
# 在小样本上测试
python core/runner/run_benchmark.py \
  --system rawllm \
  --benchmark locomo \
  --split test \
  --limit 5 \
  --model-name gpt-4o-mini
```

### 3. 查看结果

```bash
# 查看QA日志
cat results/locomo/rawllm/run_*/qa_logs.jsonl | head -1 | jq

# 查看汇总结果
cat results/locomo/rawllm/run_*/summary.json | jq
```

## 📋 当前状态

### ✅ 已完成
- [x] MemorySystem抽象接口定义
- [x] LoCoMo数据加载器
- [x] HotpotQA数据加载器
- [x] 统一评分函数
- [x] 日志记录工具
- [x] 统一评估框架
- [x] RawLLMSystem适配器（基础版本）

### ⬜ 待完成
- [ ] MemoryAgentBench数据加载器
- [ ] Mem0适配器实现
- [ ] GAM适配器实现
- [ ] LightMem适配器实现
- [ ] MIRIX适配器实现
- [ ] AGMS系统实现
- [ ] RawLLMSystem的LLM API集成

## 🔧 下一步工作

### 优先级1：完善RawLLMSystem
- [ ] 集成真实的LLM API（OpenAI/Anthropic等）
- [ ] 实现准确的token计数
- [ ] 在完整数据集上测试

### 优先级2：实现第一个真实baseline
- [ ] 研究Mem0的API
- [ ] 实现Mem0System适配器
- [ ] 在MemoryAgentBench上验证结果

### 优先级3：实现AGMS系统
- [ ] 设计L2/L3存储结构
- [ ] 实现Router决策逻辑
- [ ] 实现observe()和answer()方法

## 📊 实验运行

### 运行单个系统在单个benchmark上

```bash
python core/runner/run_benchmark.py \
  --system <system_name> \
  --benchmark <benchmark_name> \
  --split test \
  --run-id <custom_run_id> \
  --model-name <model_name>
```

### 批量运行所有系统

创建脚本 `run_all_experiments.sh`:

```bash
#!/bin/bash

SYSTEMS=("rawllm" "mem0" "gam" "lightmem" "mirix" "agms")
BENCHMARKS=("locomo" "hotpotqa" "memory_agent_bench")

for system in "${SYSTEMS[@]}"; do
  for benchmark in "${BENCHMARKS[@]}"; do
    echo "Running $system on $benchmark..."
    python core/runner/run_benchmark.py \
      --system $system \
      --benchmark $benchmark \
      --split test \
      --model-name gpt-4o-mini
  done
done
```

## 📝 注意事项

1. **统一性**: 所有系统必须使用相同的LLM backend和配置
2. **Cost记录**: 记录增量cost，不是累计cost
3. **结果验证**: 每个baseline的结果应与官方结果接近（±5%以内）
4. **日志完整性**: 确保所有字段都正确记录

## 🔍 问题排查

### 导入错误
```bash
# 确保在AGMS目录下运行
cd /share/home/qmzhu/AGMS
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 数据路径错误
检查数据文件是否存在：
```bash
ls /share/home/qmzhu/AGMS/data/locomo/locomo10.json
ls /share/home/qmzhu/AGMS/data/hotpot_qa/fullwiki/test-00000-of-00001.parquet
```

## 📚 参考文档

- `experiments.md` - 实验指导文档
- `实验复现计划.md` - 详细的分阶段计划
- Baseline参考实现：
  - GAM: `/share/home/qmzhu/AGMS/relatedwork/general-agentic-memory/eval/`
  - 官方benchmark: `/share/home/qmzhu/benchmark_github/`


