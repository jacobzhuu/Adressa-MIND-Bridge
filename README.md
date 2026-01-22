# Adressa 实体抽取→WikidataId→实体向量（NbBERT + MIND init）

本仓库脚本按 `docs/任务具体流程.md` 落地实现端到端流程：

1) 用 NbBERT(NER) 从 Adressa `news.tsv` 的标题抽取实体 span  
2) 基于 span 在 Wikidata 搜索并链接到 `WikidataId(QID)`  
3) 回填到 Adressa `news.tsv:title_entities`（MIND 同结构）  
4) 用 MIND `entity_embedding.vec` 初始化 Adressa 实体表  
5) 用挪威语标题上下文训练实体向量（InfoNCE）  
6) 导出 `entity_embedding.vec`（MIND 同格式：`Qxxx + 100维float`）

## 依赖

```bash
pip install -r requirements.txt
```

> 说明：Wikidata 查询需要联网；首次运行会下载 HuggingFace 模型权重。

## 数据目录约定

- `data/raw/adressa_one_week/`：原始 one_week 数据（输入）。
- `data/base/adressa_one_week_mind_base/`：**基线对照数据集**（MIND 格式，仅 `news.tsv`/`behaviors.tsv`）。不要在该目录上跑 01-07（避免生成 `news.with_entities.tsv`、`entity_embedding.vec`、`news.tsv.bak.*` 等副产物污染对照）。
- `data/work/adressa_one_week_mind_final/`：**实验/工作目录**（默认 `DATA_ROOT`），用于跑整条流水线并写入各类中间产物与输出。
- `data/mind/MINDsmall/`、`data/mind/MINDlarge/`：MIND 预训练实体向量与相关文件。
- `outputs/artifacts/`：中间产物与训练输出。
- `outputs/cache/`：缓存（Wikidata / HuggingFace 等）。
- `configs/`：sweep 网格配置文件。

将 one_week 转换为 MIND 格式（默认输出到 `data/base/adressa_one_week_mind_base/`）：

```bash
python scripts/convert_one_week.py --input data/raw/adressa_one_week --output data/base/adressa_one_week_mind_base
```

建议每次实验前，把 base 复制一份到工作目录再跑（任选其一）：

```bash
cp -R data/base/adressa_one_week_mind_base data/work/adressa_one_week_mind_final
# 或者用一个新的工作目录名，然后跑时显式指定 DATA_ROOT
```

## 一键脚本

### 01-03：NER → Wikidata → 写回 news.tsv

```bash
bash scripts/run_01_03.sh
```

输出日志中会包含每个 split 的 `non_empty_title_entities` 与样例实体。

> 默认 `DATA_ROOT=data/work/adressa_one_week_mind_final`；可通过 `DATA_ROOT=...` 指向你自己的工作目录（不要指向 `data/base/adressa_one_week_mind_base`）。

### 04-07：初始化 → 训练 → 导出 → 评估

使用 MINDsmall 初始化：

```bash
bash scripts/run_04_07_mindsmall.sh
```

使用 MINDlarge 初始化（train+val+test 并集初始化，覆盖率更高）：

```bash
bash scripts/run_04_07_mindlarge.sh
```

输出日志包含：

- `[COVERAGE]`：初始化命中率（unique 实体覆盖 + mention 加权覆盖）
- `[EVAL]`：val/test 上的 Recall@K 与 MRR（mention→entity 检索）

## 默认超参数（已设为最优搭配）

基于 sweep（SEEN 优先）的最优组合已写入 `scripts/run_01_03.sh` 默认值，无需额外设置即可直接跑：

```text
NER_HEURISTIC_MODE=fallback
NER_HEURISTIC_MAX_MENTIONS=4
NER_HEURISTIC_SCORE=0.35
WIKIDATA_MIN_MATCH=0.6
WIKIDATA_MIN_MATCH_HEUR=0.92
```

如需覆盖，可在命令前用环境变量传入，例如：

```bash
NER_HEURISTIC_SCORE=0.45 WIKIDATA_MIN_MATCH_HEUR=0.9 bash scripts/run_01_03.sh
```

## 超参数批量对比（SEEN 优先）

用 `scripts/sweep_01_07.py` 批量运行：

- `bash scripts/run_01_03.sh`
- `bash scripts/run_04_07_mindsmall.sh`
- `bash scripts/run_04_07_mindlarge.sh`

并把每次运行的完整日志与解析出的所有指标保存到磁盘，最后按 **SEEN 优先、FULL 次之** 自动选出最佳超参数配置。

默认网格直接跑：

```bash
python scripts/sweep_01_07.py
```

查看将要运行的组合（不执行）：

```bash
python scripts/sweep_01_07.py --dry_run
```

自定义网格（环境变量名 → 值列表）：

```bash
python scripts/sweep_01_07.py \
  --grid_json '{"WIKIDATA_MIN_MATCH_HEUR":[0.85,0.90,0.92],"NER_HEURISTIC_MAX_MENTIONS":[4,6]}'
```

使用网格文件：

```bash
python scripts/sweep_01_07.py --grid_file configs/sweep_grid.seen_first.json
```

跑满 100 次（20 组配置 × repeat=5）：

```bash
python scripts/sweep_01_07.py \
  --grid_file configs/sweep_grid_100runs.seen_first.json \
  --repeat 5
```

常用选项：

- `--echo`：把子脚本原始输出同步打印到屏幕（默认只写入 run 的 log 文件）
- `--no_color`：关闭彩色输出（例如保存终端日志时）

输出目录（可溯源）在 `outputs/artifacts/sweeps/<timestamp>/`，包括每次 run 的 `logs/`、`metrics.json`、以及汇总的 `summary.csv` / `config_summary.csv` / `best_config.json`。

补充说明：

- sweep 会在每个 run 下创建隔离的 `data_root/`（复制 `--base_data_root` 下的 `news.tsv`），避免并发/多次运行时互相覆盖主数据目录。
- 每次 run 的完整日志在 `outputs/artifacts/sweeps/<timestamp>/runs/<run_id>/logs/`。

## 结果复盘（可选）

如果只想对某次 sweep 目录下的现有日志重新解析并生成汇总（不重新训练），可用：

```bash
python scripts/reanalyze_sweep.py outputs/artifacts/sweeps/<timestamp>
```

## 1) NER：标题实体 span

以 train 为例（val/test 同理）：

```bash
python scripts/01_ner_titles_nbbert.py \
  --news_tsv data/work/adressa_one_week_mind_final/train/news.tsv \
  --output_jsonl outputs/artifacts/train.mentions.jsonl
```

## 2) Wikidata 链接：span → QID

```bash
python scripts/02_link_to_wikidata.py \
  --mentions_jsonl outputs/artifacts/train.mentions.jsonl \
  --output_jsonl outputs/artifacts/train.linked.jsonl \
  --cache_db outputs/cache/wikidata_search.sqlite \
  --lang nb \
  --sleep 0.1
```

常见问题：

- 如果你机器上配置了系统代理导致 `ProxyError`，可加 `--no-trust-env` 让 requests 忽略系统/环境代理。
- 运行中断后可用 `--resume` 继续（脚本会按已写出的行数跳过输入并追加输出）。

## 3) 回填 news.tsv：写入 title_entities

```bash
python scripts/03_write_title_entities_to_tsv.py \
  --news_tsv data/work/adressa_one_week_mind_final/train/news.tsv \
  --linked_jsonl outputs/artifacts/train.linked.jsonl \
  --output_tsv data/work/adressa_one_week_mind_final/train/news.with_entities.tsv
```

## 4) 构建实体词表 + 用 MIND 初始化

把各 split 生成的 `news.with_entities.tsv` 一起喂进去（没有 `val/` 也可以，只传存在的 split）：

```bash
python scripts/04_build_entity_vocab_and_init.py \
  --news_tsv \
    data/work/adressa_one_week_mind_final/train/news.with_entities.tsv \
    data/work/adressa_one_week_mind_final/test/news.with_entities.tsv \
  --mind_entity_vec data/mind/MINDsmall/train/entity_embedding.vec \
  --output_dir outputs/artifacts/entities
```

产物：

- `outputs/artifacts/entities/entity_vocab.txt`
- `outputs/artifacts/entities/entity_init.npy`
- `outputs/artifacts/entities/entity_init.vec`

## 5) 训练挪威语实体向量（InfoNCE）

```bash
python scripts/05_train_entity_embeddings_no.py \
  --news_tsv data/work/adressa_one_week_mind_final/train/news.with_entities.tsv \
  --entity_vocab outputs/artifacts/entities/entity_vocab.txt \
  --entity_init outputs/artifacts/entities/entity_init.npy \
  --output_dir outputs/artifacts/no_entity_train
```

## 6) 导出 entity_embedding.vec（MIND 格式）

```bash
python scripts/06_export_entity_embedding_vec.py \
  --entity_vocab outputs/artifacts/entities/entity_vocab.txt \
  --entity_matrix outputs/artifacts/no_entity_train/entity_trained.npy \
  --output_vec data/work/adressa_one_week_mind_final/train/entity_embedding.vec
```
