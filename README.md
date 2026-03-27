# Video RL 子能力数据集构建工具

围绕 **11 个视频子能力** 构建统一的 Video RL/RLVR 训练数据集，支持从原始数据下载到最终 RL-ready JSONL 的全链路自动化。

---

## 关键设计决策

### 1. 去除纯评测 Benchmark（防止数据污染）

以下 12 个数据集为纯评测基准，**严禁**用作训练数据，已在 `registry/benchmark_holdout.yaml` 中注册并由代码自动拦截：

| 已去除的 Benchmark | 原因 |
|---|---|
| TempCompass / TemporalBench / MVBench | 时序理解评测集 |
| MLVU / LongVideoBench / Video-MME | 长视频/多模态评测集 |
| EgoSchema / EgoIntent | 自中心评测集 |
| Perception Test / Video-MMMU / VSI-Bench | 感知/知识/空间评测集 |
| TutorialVQA | 教程视频评测集 |

### 2. 视频存储：本地路径引用，不使用 base64

视频文件保存在磁盘 `data/raw/<数据集>/videos/`，JSONL 中通过 `file://` URI 引用：

```json
"video_url": {"url": "file:///abs/path/to/video.mp4", "clip_fps": 1.0}
```

JSONL 文件保持轻量（~数百 MB），视频 I/O 交由训练框架的 DataLoader 处理。

### 3. 统一使用 hf_dl.sh 下载

所有 HuggingFace 数据集通过自定义脚本下载，路径在 `configs/download_config.yaml` 中配置：

```bash
bash hf_dl.sh <repo_id> dataset \
    --browser-executable-path "/root/data/home/zhangyuxiang/utils/dl_hf/chrome-linux/chrome" \
    --scrape-concurrency 4
```

非 HF 托管的数据集（如 Ego4D、Something-Something V2）会生成手动下载说明文档。

---

## 11 个子能力及数据源

| 编号 | 子能力 | 稳态采样比 | Priority-1 数据源 |
|---|---|---|---|
| A | temporal_atomic（时序原子事件） | 14% | Video-R1-260k |
| B | temporal_count_order（计数与排序） | 9% | Video-R1-260k, FineGym, COIN |
| C | temporal_grounding（时序定位） | 11% | QVHighlights, ActivityNet Captions |
| D | long_video_retrieval_memory（长视频检索记忆） | 11% | TVQA, Ego4D, ActivityNet Captions |
| E | causal_relation_reasoning（因果关系推理） | 9% | NExT-QA, STAR, CLEVRER |
| F | future_event_counterfactual（未来事件/反事实） | 10% | NEP V1-33K, STAR, CLEVRER |
| G | egocentric_intent_next_step（自中心意图/下一步） | 8% | Ego4D, EPIC-KITCHENS |
| H | spatial_spatiotemporal_reasoning（空间/时空推理） | 10% | SpaceR-151k, CLEVRER, AGQA |
| I | multimodal_av_fusion（多模态音视频融合） | 6% | AVQA, MUSIC-AVQA, Ego4D |
| J | topic_plot_knowledge_acquisition（主题/知识获取） | 6% | TVQA, ActivityNet, YouCook2 |
| K | anti_shortcut_contrast（反捷径对比） | 6% | Video-R1, SpaceR, TPO/TEMPO 构造 |

---

## 数据集三层分类

### Layer A：RL 原生数据集（直接可用）

| 数据集 | HuggingFace Repo | 规模 | 用途 |
|---|---|---|---|
| Video-R1-260k | `Video-R1/Video-R1-data` | 260k | 时序/因果 RL 训练 |
| Video-R1-CoT-165k | `Video-R1/Video-R1-data` | 165k | SFT 冷启动（含 CoT） |
| SpaceR-151k | `RUBBISHLIKE/SpaceR-151k` | 151k | 空间推理 RL 训练 |
| NEP V1-33K | `haonan3/V1-33K` | 33k | 下一事件预测 |

### Layer B/C：可训练数据集（仅使用 train split）

Priority-1（已实现完整 adapter）：
- NExT-QA、STAR、CLEVRER、QVHighlights、TVQA、ActivityNet Captions、Ego4D

Priority-2（待实现）：
- VLEP、AGQA、ANetQA、COIN、CrossTask、Assembly101、SSV2、FineGym、AVQA、MUSIC-AVQA、YouCook2、EPIC-KITCHENS

Priority-3（待实现）：
- Diving48、Jester、FineDiving、TACoS、Charades-STA、AVSD、Ego-Exo4D、Action Genome

---

## 项目结构

```
Sub-capability_Video/
├── pyproject.toml                      # Python 依赖
├── .gitignore
│
├── registry/                           # 数据集注册与安全检查
│   ├── benchmark_holdout.yaml          # 12 个禁用 benchmark
│   ├── capability_map.yaml             # 11 子能力 -> 数据源映射
│   └── __init__.py                     # is_benchmark(), assert_valid_split()
│
├── schema/
│   ├── canonical.py                    # Pydantic 数据模型（CanonicalSample 等）
│   └── validators.py                   # JSONL 校验 + 视频文件存在性检查
│
├── configs/
│   ├── download_config.yaml            # hf_dl.sh 路径配置
│   ├── dataset_registry.yaml           # 数据集总目录
│   ├── reward_templates.yaml           # 11 个奖励模板权重
│   └── sampling_mixture.yaml           # 采样比例 + 课程学习调度
│
├── source_adapters/                    # 每个原始数据集一个 adapter
│   ├── base_adapter.py                 # 基类 ABC
│   ├── video_r1.py                     # Video-R1-260k
│   ├── spacer.py                       # SpaceR-151k
│   ├── nep_v1.py                       # NEP V1-33K
│   ├── nextqa.py / star.py / clevrer.py
│   ├── qvhighlights.py / tvqa.py
│   ├── activitynet_captions.py / ego4d.py
│   └── ...
│
├── builders/                           # 每个子能力一个 builder
│   ├── base.py                         # 基类（含 benchmark 守卫）
│   ├── a_temporal_atomic/              # build.py + download.py
│   ├── b_temporal_count_order/
│   ├── ... (c ~ k)
│   └── k_anti_shortcut_contrast/
│
├── graders/
│   ├── ruler_graders.py                # 11 个规则评分器
│   ├── model_graders.py                # 模型评分器（仅兜底）
│   └── reward_templates.py             # 奖励计算函数
│
├── pair_constructors/
│   ├── tpo_pairs.py                    # TPO 偏好对构造
│   └── tempo_pairs.py                  # TEMPO 时序扰动对构造
│
├── pipeline/
│   ├── download_all.py                 # 生成统一下载脚本
│   ├── build_all.py                    # 运行所有 builder
│   ├── dedup.py                        # 跨子能力去重
│   ├── merge.py                        # 合并最终 JSONL
│   ├── validate.py                     # Schema + 文件校验
│   └── stats.py                        # 成功标准检查
│
├── tests/                              # 33 个测试用例
│   ├── test_schema.py
│   ├── test_benchmark_guard.py
│   └── test_adapters.py
│
└── data/                               # .gitignore，不提交
    ├── raw/<数据集>/videos/             # 原始视频
    ├── raw/<数据集>/annotations/        # 原始标注
    ├── canonical/<子能力>.jsonl          # 标准化输出
    ├── rl_ready/<子能力>_rl.jsonl        # RL-ready 输出
    └── merged/                          # 合并后的最终文件
```

---

## 快速开始

### 安装依赖

```bash
pip install pydantic pyyaml fire tqdm orjsonl pytest
```

### 运行测试

```bash
python -m pytest tests/ -v
```

### 五步构建流程

```bash
# 1. 生成下载脚本（仅 Priority-1 数据集）
python -m pipeline.download_all --priority 1

# 2. 执行下载
bash data/download_all.sh

# 3. 构建标准化 JSONL
python -m pipeline.build_all

# 4. 去重 + 合并
python -m pipeline.dedup
python -m pipeline.merge

# 5. 校验 + 统计
python -m pipeline.validate
python -m pipeline.stats
```

---

## 输出 Schema 示例

每条样本的 JSON 结构（顶层 4 个键：`messages`、`graders`、`data_info`、`extra_info`）：

```json
{
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "You are a video reasoning assistant..."}]},
    {"role": "user", "content": [
      {"type": "video_url", "video_url": {"url": "file:///data/raw/nextqa/videos/001.mp4", "clip_fps": 1.0}},
      {"type": "text", "text": "Question: Which action happens first? A. sit B. stand. Answer with one capital letter."}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "A"}]}
  ],
  "graders": [{"type": "ruler", "name": "option_match", "gt": "A", "params": {"min_score": 1.0}}],
  "data_info": {
    "data_id": "nextqa_q001",
    "ability": "Causal",
    "datasource": "NExT-QA",
    "task_type": "mcq",
    "sub_ability": ["causal_why_how"],
    "build_info": {"build_type": "converted"}
  },
  "extra_info": {
    "reward_info": {"reward_template": "causal_relation_v1", "weights": {"ans": 0.6, "logic": 0.2, "support": 0.1, "format": 0.1}},
    "sampling_info": {"mix_bucket": "causal_relation_reasoning", "curriculum_stage": "main"}
  }
}
```

---

## 11 个奖励模板

| 模板 | 公式 |
|---|---|
| temporal_atomic_v1 | `R = 0.7*ans + 0.2*temporal + 0.1*format` |
| count_order_v1 | `R = 0.6*ans + 0.2*order_or_count + 0.1*support + 0.1*format` |
| grounding_v1 | `R = 0.45*ans + 0.40*span_iou + 0.10*contrast + 0.05*format` |
| long_retrieval_v1 | `R = 0.35*ans + 0.35*retrieval + 0.20*support + 0.10*efficiency` |
| causal_relation_v1 | `R = 0.6*ans + 0.2*logic + 0.1*support + 0.1*format` |
| future_pred_v1 | `R = 0.5*ans + 0.2*future_slot + 0.2*contrast + 0.1*format` |
| ego_intent_v1 | `R = 0.35*what + 0.30*why + 0.25*next + 0.10*format` |
| spatial_video_v1 | `R = 0.5*ans + 0.25*spatial + 0.15*support + 0.10*format` |
| av_fusion_v1 | `R = 0.55*ans + 0.20*cross_modal + 0.15*ablation + 0.10*format` |
| knowledge_acq_v1 | `R = 0.5*ans + 0.2*support + 0.2*adaptation + 0.1*format` |
| anti_shortcut_v1 | `R = 0.45*clean_ans + 0.35*perturbed_ans + 0.20*contrast_gap` |

---

## 课程学习调度

| 阶段 | 训练步数占比 | 重点子能力 |
|---|---|---|
| Warmup | 前 20% | temporal_atomic, temporal_count_order, anti_shortcut |
| Main | 中间 50% | temporal_grounding, long_retrieval, causal, future_prediction |
| Refinement | 最后 30% | multimodal_av_fusion, topic_plot_knowledge |

---

## 成功标准

| 指标 | 目标 |
|---|---|
| Canonical 样本总量 | >= 200,000 |
| RL-ready 可验证样本 | >= 80,000 |
| 带 support_spans 的样本占比 | >= 30% |
| 每个子能力独立数据源数 | >= 3 |
| 时序/空间桶的 text-only pass rate | 尽可能低 |
| 所有样本通过 schema 校验 + 视频文件存在性检查 | 100% |

---

## 当前实现状态

- 77 个源文件，33/33 测试通过
- 10 个 Priority-1 source adapter 已完整实现
- 11 个子能力 builder 已搭建完成
- 11 个 ruler grader + 11 个 reward template 已实现
- Pipeline 全链路脚本就绪（download → build → dedup → merge → validate → stats）
- Priority-2/3 adapter 待后续补充
