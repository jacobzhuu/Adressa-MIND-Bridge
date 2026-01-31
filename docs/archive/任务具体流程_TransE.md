# （归档）任务具体流程（TransE 路线旧稿）

> 注意：本文档为早期过程记录，可能与当前仓库代码/脚本路径不一致；请以仓库根目录 `README.md` 为准。

# 任务具体流程（TransE 路线）— 详细修改方案（基于当前仓库代码）

> 目标：在本仓库已跑通的 Step 01-04（NER→EL→回填→MIND 初始化）基础上，把训练步骤从 `scripts/05_train_entity_embeddings_no.py`（InfoNCE）改为 **Wikidata 子图上的 TransE**，最终仍导出 `entity_embedding.vec`（与 MIND 同格式：`Qxxx + 100维`）。

---

## 0. 你现在拥有什么（当前仓库现状）

### 0.1 已经可直接复用的脚本（无需改动）
- 数据准备：`scripts/convert_one_week.py`（raw one_week → MIND 格式目录）
- Step 01：`scripts/01_ner_titles_nbbert.py`（标题 NER → `*.mentions.jsonl`）
- Step 02：`scripts/02_link_to_wikidata.py`（Wikidata 搜索链接 → `*.linked.jsonl`）
- Step 03：`scripts/03_write_title_entities_to_tsv.py`（回填 `title_entities` → `news.tsv`）
- Step 04：`scripts/04_build_entity_vocab_and_init.py`（从 `news.tsv` 抽 `E_adressa` 并用 MIND 初始化）
- 导出：`scripts/06_export_entity_embedding_vec.py`（把 `*.npy` 矩阵导出为 `entity_embedding.vec`）
- 诊断：`scripts/08_diagnose_qid_issues.py`（QID 格式/覆盖率/EL 质量排查）
- 一键跑 Step 01-03：`scripts/run_01_03.sh`（会把 `news.with_entities.tsv` 覆盖回 `news.tsv` 并备份）

### 0.2 当前“训练”脚本为什么要替换？
`scripts/05_train_entity_embeddings_no.py` 是 **NbBERT mention→entity 对比学习**（InfoNCE），它并不使用 KG 三元组，也不满足“导师要求使用 TransE”。

---

## 1. 最终产物（验收口径）

你需要产出两类文件：
1) `DATA_ROOT/<split>/news.tsv`：包含 `title_entities`（MIND 风格 JSON list）  
2) `DATA_ROOT/<split>/entity_embedding.vec`：每行 `Qxxx` + 100 维 float（tab 分隔；和 `data/mind/.../entity_embedding.vec` 一致）

> 说明：本仓库现有导出脚本输出为 tab 分隔；MIND 的 vec 也是用 tab 分隔（仓库内文件可直接对照）。

---

## 2. 本次要改什么（相对现有流水线）

### 2.1 不变的部分（直接照跑）
- Step 00（可选）：raw 数据 → MIND 格式目录
- Step 01-03：NER → Entity Linking → 回填 `news.tsv:title_entities`
- Step 04：构建 `entity_vocab.txt` + `entity_init.npy`（MIND 初始化）
- Step 06：导出 `entity_embedding.vec`

### 2.2 需要新增的部分（TransE 必需）
需要新增 2 个 Python 脚本（建议编号从 09 起，避免破坏已有 sweep/实验），并遵守下面这些“可复现、可跑通”的规范（不额外引入依赖，保持与当前仓库风格一致）。

> 当前项目的现实约束：`E_adressa` 规模在几千级，且 MIND 覆盖率很低（你之前跑的 one_week 里 MIND 命中约 8%~10%）。因此新增步骤必须默认采用 **Inductive / Anchor-locked** 的 TransE：冻结锚点实体 + 冻结关系向量，避免坐标系漂移。

#### 2.2.1 `scripts/09_fetch_wikidata_triples.py`（Wikidata 子图构建）

**目标**：从 Step 04 的 `entity_vocab.txt`（`E_seed`）出发，抓取 Wikidata 中与这些实体相关的三元组，生成一个“足够小但有用”的训练子图。

**输入（必须）**：
- `entity_vocab.txt`：Step 04 输出（1 行 1 个 QID），作为种子实体集合 `E_seed`。
- `relation_embedding.vec`：建议用 `data/mind/MINDlarge/train/relation_embedding.vec`，读取允许关系集合 `R_mind`（PID 集）。

**输出（必须）**：
- `kg_triples.txt`：每行 3 列 `head  relation  tail`（QID/PID/QID；空格或 tab 分隔均可，但建议用 tab）。  
  - 仅包含 item→item（`Q...`→`Q...`）三元组；不要写 literal（string/number/time/geo 等）。
  - 必须去重（同一行完全一致只保留 1 次）。
- `kg_stats.json`：必须包含可解释统计，至少包括：
  - `num_seed_entities`、`num_seed_with_edges`、`seed_edge_coverage`
  - `num_relations_kept`、`num_triples`（去重后）、`avg_triples_per_seed`
  - `top_relations`（按出现次数 topN，便于判断是不是只有 P31/P279）

**输出（可选，但建议）**：
- `kg_entities.txt`：若你决定把邻居实体也纳入训练（见下文），则写出“训练图里出现过的全部实体（seed+neighbor）”的去重列表。

**抽取与剪枝规范（必须）**：
- 关系限制：只保留 `pid ∈ R_mind` 的边（确保后续可以用 MIND 的 `relation_embedding.vec` 初始化并冻结关系向量）。
- Top-K：必须提供 `--max_triples_per_entity`（或同等效果的剪枝），限制每个 seed 最多保留 K 条 statement（否则 triples 很容易爆炸）。
- 逆关系：为了让实体既能当 head 也能当 tail、但又不需要做 SPARQL 反查入边，必须对每条 `(h, pid, t)` 额外生成一条 `(t, pid_inv, h)`：
  - `pid_inv` 命名规范：`{pid}_inv`（例如 `P31_inv`）；不要与原 PID 冲突。
  - `pid_inv` 不需要出现在 MIND 的 relation vec；训练脚本中用 `r(pid_inv) = -r(pid)` 生成并冻结即可。
- 邻居策略（二选一，必须明确并在 stats 里记录）：
  1) **Seed-only（默认推荐）**：只保留 `tail ∈ E_seed` 的边（训练不需要额外实体表，最稳最简单）；缺点是 triples 可能偏少。  
  2) **Seed+Neighbor（增强版）**：允许 `tail ∉ E_seed`，把这些邻居实体加入训练图（写 `kg_entities.txt`）；缺点是需要在训练脚本里为 neighbor 建 embedding（可随机 init，不导出），并更容易出现“无锚点连通分量”。

**联网与可用性规范（必须）**：
- 数据源：优先用 Wikidata API `wbgetentities`（支持批量请求多个 QID）获取 `claims`；不要默认用 SPARQL/dump（工程成本过高）。
- 速率控制：必须支持 `--sleep`（未命中缓存时 sleep），并支持重试退避（参考 `src/adressa_entity/wikidata.py` 的 backoff 风格）。
- 缓存与断点：必须支持本地缓存（建议 sqlite，路径默认放 `outputs/cache/`），并提供 `--resume`（或等价方案）避免中断后重跑全部 QID。
- 代理兼容：建议提供 `--trust-env/--no-trust-env`（与 `scripts/02_link_to_wikidata.py` 一致），避免用户系统代理导致请求失败。

#### 2.2.2 `scripts/10_train_transe.py`（Inductive TransE 训练）

**目标**：在不破坏 MIND 坐标系的前提下，用 `kg_triples.txt` 训练/微调 Adressa 的实体向量，输出与 `entity_vocab.txt` 行对齐的 `entity_trained.npy`。

**输入（必须）**：
- `kg_triples.txt`：来自 09 脚本。
- `entity_vocab.txt`：来自 Step 04（决定最终导出的实体顺序与数量）。
- `entity_init.npy` + `entity_init_mask.npy`：来自 Step 04（用于初始化与锚点冻结）。
- `relation_embedding.vec`：来自 MIND（PID→100d 向量）。

**输出（必须）**：
- `entity_trained.npy`：shape 必须是 `(len(entity_vocab), 100)`，顺序必须与 `entity_vocab.txt` 完全一致（后续 `scripts/06_export_entity_embedding_vec.py` 依赖这一点）。
- `train_log.txt`：至少记录每个 epoch 的 loss（以及可选的 coverage/采样统计）。
- `config.json`：记录训练超参、输入文件 sha1/mtime 等关键信息，保证可复现。

**训练与对齐规范（必须）**：
- 维度：实体/关系维度固定 100（与 MIND vec 一致）。
- 关系初始化与冻结：
  - `pid` 的关系向量从 MIND `relation_embedding.vec` 读取并冻结（不更新）。
  - `pid_inv` 的关系向量用 `-r(pid)` 构造并冻结（不更新）。
- 锚点实体冻结（核心）：
  - 以 `entity_init_mask.npy==1` 定义 `E_anchor = E_seed ∩ E_mind`；这些行必须冻结（不更新）。
  - 训练结束后必须做一致性检查：锚点行的向量应保持不变（或变化 < 极小阈值），否则视为训练失败/实现不符合要求。
- 负采样：按 TransE 常规做 head/tail corruption；必须避免采到原三元组（至少避免采到同一个实体导致无效负例）。
- 稳定性约束：必须提供至少一种抑制平移自由度的手段（例如 entity norm clipping / weight decay），并默认开启一个“能跑通”的配置。

**数据覆盖率规范（必须）**：
- 必须在训练开始前/后输出（并写入 log 或 stats）：
  - `num_seed_with_triples`、`seed_triple_coverage`
  - `num_anchor_entities`、`num_anchor_with_triples`
  - （如果启用 neighbor）`num_neighbor_entities`、`num_entities_total`
- 若 `kg_triples.txt` 为空或覆盖率极低：脚本必须能 fallback（直接输出 `entity_init.npy` 作为 `entity_trained.npy`，并在日志中明确提示原因）。

（可选但推荐）**集成规范**：
- 增加一个一键脚本 `scripts/run_05_06_transe_mindlarge.sh`（或类似命名），行为对齐现有 `run_04_07_*.sh`：自动组织路径、写入 `outputs/artifacts/`、并把 `entity_embedding.vec` 复制到各 split。这样导师验收时能“一条命令跑通”。

---

## 3. 从 0 到 1：基于当前代码需要执行的全部步骤（含命令）

下面按“你真的要跑出来结果”的顺序写，直接照着执行即可。

### Step 0：环境与目录（一次性）

安装依赖：

```bash
pip install -r requirements.txt
```

确定你的工作目录（建议不要污染 base，对照 README 的约定）：
- `DATA_ROOT=data/work/adressa_one_week_mind_final`（默认）
- `ARTIFACTS_DIR=outputs/artifacts`
- `CACHE_DIR=outputs/cache`

> NER/EL 需要联网（Wikidata + HuggingFace 模型下载）；`scripts/run_01_03.sh` 会自动设置 `HF_HOME` 到缓存目录。

---

### Step 0.1（可选）：raw one_week → MIND 格式目录

如果你手上还是 raw JSON：

```bash
python scripts/convert_one_week.py \
  --input data/raw/adressa_one_week \
  --output data/base/adressa_one_week_mind_base

cp -R data/base/adressa_one_week_mind_base data/work/adressa_one_week_mind_final
```

> `convert_one_week.py` 默认 `--split 5,1,1`（会生成 `train/val/test`）。如果你不想要 `val/`，可以改成 `--split 5,0,1`。

---

### Step 1：Step 01-03（NER → Entity Linking → 回填 news.tsv）

直接用现成一键脚本（推荐）：

```bash
DATA_ROOT=data/work/adressa_one_week_mind_final \
ARTIFACTS_DIR=outputs/artifacts \
CACHE_DIR=outputs/cache \
bash scripts/run_01_03.sh
```

执行后你应该看到：
- `DATA_ROOT/<split>/news.tsv` 已被覆盖（旧文件备份为 `news.tsv.bak.<timestamp>`）
- `ARTIFACTS_DIR/<split>.mentions.jsonl`、`ARTIFACTS_DIR/<split>.linked.jsonl`

如果想先排查 QID/链接质量（强烈建议至少跑一次）：

```bash
python scripts/08_diagnose_qid_issues.py \
  --data_root data/work/adressa_one_week_mind_final \
  --mind_entity_vec data/mind/MINDlarge/train/entity_embedding.vec \
  --artifacts_dir outputs/artifacts
```

---

### Step 2：Step 04（构建实体词表 + MIND 初始化）

这一步用现成脚本，但建议用 MINDlarge 的 train/val/test 三个 vec 补齐覆盖率。  
如果你 **有** `val/` split，就把 `val/news.tsv` 也加到 `--news_tsv` 列表里；如果你没有 `val/` split，可忽略（或用下面的 bash 自动探测）。

```bash
python scripts/04_build_entity_vocab_and_init.py \
  --news_tsv \
    data/work/adressa_one_week_mind_final/train/news.tsv \
    data/work/adressa_one_week_mind_final/test/news.tsv \
  --mind_entity_vec \
    data/mind/MINDlarge/train/entity_embedding.vec \
    data/mind/MINDlarge/val/entity_embedding.vec \
    data/mind/MINDlarge/test/entity_embedding.vec \
  --output_dir outputs/artifacts/entities_mindlarge
```

（可选）自动探测 splits 的版本：

```bash
DATA_ROOT=data/work/adressa_one_week_mind_final
news_args=()
for sp in train val test; do
  if [[ -f "$DATA_ROOT/$sp/news.tsv" ]]; then
    news_args+=("$DATA_ROOT/$sp/news.tsv")
  fi
done

python scripts/04_build_entity_vocab_and_init.py \
  --news_tsv "${news_args[@]}" \
  --mind_entity_vec \
    data/mind/MINDlarge/train/entity_embedding.vec \
    data/mind/MINDlarge/val/entity_embedding.vec \
    data/mind/MINDlarge/test/entity_embedding.vec \
  --output_dir outputs/artifacts/entities_mindlarge
```

检查产物是否存在：
- `outputs/artifacts/entities_mindlarge/entity_vocab.txt`
- `outputs/artifacts/entities_mindlarge/entity_init.npy`
- `outputs/artifacts/entities_mindlarge/entity_init_mask.npy`

---

### Step 3：新增 Step 05（抓取 Wikidata 子图三元组）

先实现 `scripts/09_fetch_wikidata_triples.py`（见第 4 节“新增脚本规格”），实现后执行：

```bash
python scripts/09_fetch_wikidata_triples.py \
  --seed_entity_vocab outputs/artifacts/entities_mindlarge/entity_vocab.txt \
  --mind_relation_vec data/mind/MINDlarge/train/relation_embedding.vec \
  --output_dir outputs/artifacts/wikidata_subgraph \
  --lang nb \
  --sleep 0.1
```

你应得到：
- `outputs/artifacts/wikidata_subgraph/kg_triples.txt`
- `outputs/artifacts/wikidata_subgraph/kg_stats.json`
- （可选）`outputs/artifacts/wikidata_subgraph/kg_entities.txt`（如果你选择把邻居实体加入训练词表）

> 建议：第一版先做“能跑通”的子图：只保留 `pid ∈ R_mind`，并设置每个实体最多保留 Top-K 条 statement（否则 triples 会爆炸）。

---

### Step 4：新增 Step 06（训练 TransE）

先实现 `scripts/10_train_transe.py`（见第 4 节），实现后执行：

```bash
python scripts/10_train_transe.py \
  --kg_triples outputs/artifacts/wikidata_subgraph/kg_triples.txt \
  --seed_entity_vocab outputs/artifacts/entities_mindlarge/entity_vocab.txt \
  --entity_init outputs/artifacts/entities_mindlarge/entity_init.npy \
  --entity_init_mask outputs/artifacts/entities_mindlarge/entity_init_mask.npy \
  --mind_relation_vec data/mind/MINDlarge/train/relation_embedding.vec \
  --output_dir outputs/artifacts/transe_train \
  --device mps
```

你应得到：
- `outputs/artifacts/transe_train/entity_trained.npy`（行顺序与 `seed_entity_vocab` 对齐）
- `outputs/artifacts/transe_train/train_log.txt`（loss 曲线/统计）

必做 sanity check（避免“训练把锚点带跑偏”）：
- 如果你选择冻结：`entity_init_mask==1` 的行在 `entity_trained.npy` 中应保持不变（或变化极小）。

---

### Step 5：导出 entity_embedding.vec（沿用现有导出脚本）

```bash
python scripts/06_export_entity_embedding_vec.py \
  --entity_vocab outputs/artifacts/entities_mindlarge/entity_vocab.txt \
  --entity_matrix outputs/artifacts/transe_train/entity_trained.npy \
  --output_vec outputs/artifacts/transe_train/entity_embedding.vec
```

把该 vec 复制到各 split（与现有 `run_04_07_*.sh` 的行为保持一致）：

```bash
DATA_ROOT=data/work/adressa_one_week_mind_final
for sp in train val test; do
  if [[ -f "$DATA_ROOT/$sp/news.tsv" ]]; then
    cp outputs/artifacts/transe_train/entity_embedding.vec "$DATA_ROOT/$sp/entity_embedding.vec"
  fi
done
```

---

### Step 6：评估/验收（TransE 路线的正确打开方式）

现有的 `scripts/07_eval_entity_embedding_retrieval.py` 依赖 NB-BERT mention + 投影层，适配的是 InfoNCE，不适配纯 TransE。

TransE 路线建议你至少做两类验收：
1) **KG 内部 link prediction**：在 `kg_triples.txt` 上划分 train/valid/test，计算 MRR / Hits@K  
2) **对齐稳定性**：检查锚点实体是否保持不变；以及新实体是否能连到锚点（例如看新实体的最近邻是否出现合理的锚点实体）

> 如果你后续还要接推荐模型，下游 A/B 才是最终指标：直接替换实体向量输入，观察推荐指标变化。

---

## 4. 必须新增的脚本（规格写清楚，照着实现即可）

### 4.1 `scripts/09_fetch_wikidata_triples.py`（Wikidata → `kg_triples.txt`）

**目标**：从种子实体 `E_seed`（`entity_vocab.txt`）出发，抓取 Wikidata claims，生成可训练的三元组。

建议命令行参数（最少要有这些）：
- `--seed_entity_vocab`: `entity_vocab.txt`
- `--mind_relation_vec`: MIND `relation_embedding.vec`（用于得到允许的 PID 集合 `R_mind`）
- `--output_dir`
- `--sleep` / `--max_retries` / `--timeout`（参考现有 WikidataSearcher 的风格）
- `--max_triples_per_entity`（防爆）
- `--keep_neighbors`（是否把 tail 邻居加入 `kg_entities.txt`，并在 `kg_stats.json` 里记录）

推荐实现策略（可落地，不要上来就 SPARQL / dump）：
1) 调 `wbgetentities`（一次多个 QID）获取 `claims`  
2) 只保留 `datavalue.type == 'wikibase-entityid'` 且目标是 item 的 statement  
3) 只保留 `pid ∈ R_mind`  
4) 写出三元组：`head pid tail`  
5) 额外写逆关系三元组：`tail pid_inv head`（`pid_inv` 形如 `P31_inv`），并在 stats 里记录逆关系数量

> 逆关系向量怎么处理：训练脚本里把 `r(P31_inv) = -r(P31)`（并冻结）即可，不需要在抓取阶段生成 relation vec。

---

### 4.2 `scripts/10_train_transe.py`（TransE 训练）

**目标**：在 MIND 坐标系里“镶嵌” Adressa 实体向量，输出与 `seed_entity_vocab` 对齐的 `entity_trained.npy`。

建议命令行参数（最少要有这些）：
- `--kg_triples`
- `--seed_entity_vocab`（决定最终要导出的实体集合与顺序）
- `--entity_init` / `--entity_init_mask`（从 Step 04 来，用于初始化与锚点冻结）
- `--mind_relation_vec`（PID→向量，冻结）
- `--output_dir`
- 训练超参：`--epochs --lr --margin --batch_size --neg_ratio --weight_decay --device`

训练细节（保证“对齐不漂移”）：
- **冻结锚点实体**：`entity_init_mask==1` 的行不更新（可通过梯度 mask 实现）
- **冻结关系向量**：`R_mind` 不更新；逆关系用 `-r` 且同样冻结
- 负采样：对每条正样本随机替换 head 或 tail（避免采到相同实体）
- 可选：每步对实体向量做 norm clipping（例如限制到 1.0），抑制平移自由度

输出文件建议：
- `entity_trained.npy`（N_seed×100）
- `train_log.txt`（loss/学习率/采样统计）
- `config.json`（记录超参，便于复现）

---

## 5. 常见坑（提前写进文档，减少返工）

1) **三元组太少**：很多实体在 Wikidata 很稀疏。对策：放宽 `max_triples_per_entity`、扩大允许的 PID（但仍建议限制在 `R_mind` 内），必要时对低度实体做 2-hop（要控规模）。  
2) **孤立连通分量无锚点**：会导致整体漂移。对策：数据侧只保留能连到锚点的边；或接受“该分量保持初始化”并在报告解释。  
3) **把锚点训练跑偏**：最致命。对策：锚点直接冻结，并做导出前差分检查。  
4) **Wikidata 限流/不稳定**：对策：批量请求、缓存、重试退避、支持断点续跑。  
