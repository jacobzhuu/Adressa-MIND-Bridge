# 项目性能优化分析报告

通过对项目核心脚本的代码审计，我发现了以下几个关键的性能优化点。按照影响程度和优先级排序如下：

## 1. 核心瓶颈：I/O 与网络 (High Priority)

这是目前导致脚本运行缓慢（Stalling）或卡顿的最主要原因，主要集中在与外部 API 交互和数据库写入的环节。

### 实体链接 (`scripts/steps/02_link_to_wikidata.py`)

*   **问题 A：串行网络请求 (严重)**
    *   **现状**：脚本使用单线程循环调用 `searcher.best_candidate()`，每次只处理一个 mention。虽然有缓存，但对于数百万未命中的实体，这意味着数百万次串行的 HTTP 请求。
    *   **优化**：引入 `ThreadPoolExecutor` 对 `searcher.best_candidate` 进行并发调用。
    *   **预期提升**：网络吞吐量提升 5-10 倍。

*   **问题 B：高频 SQLite 提交 (严重)**
    *   **现状**：`WikidataSearcher.search` 每次查询后都会执行 `self._conn.commit()`。SQLite 的 commit 涉及磁盘同步，极其耗时。
    *   **优化**：实现批量提交机制（例如每 100 次查询或每批次提交一次），或者使用 `executemany`。

*   **问题 C：高频文件刷新**
    *   **现状**：每写入一行 JSONL 结果都调用 `f_out.flush()`。这会频繁打断系统的磁盘 I/O 优化。
    *   **优化**：移除 `flush()`，让操作系统自行管理缓冲区，或者每 N 行手动 flush 一次。

### 知识图谱抓取 (`scripts/transe/09_fetch_wikidata_triples.py`)

*   **问题：批处理中的高频提交**
    *   **现状**：虽然该脚本已经使用了 `batch_size` 进行 `wbgetentities` 请求（很好），但在处理返回结果的 `_flush_pending` 函数中，它对**每一个**实体都单独调用数据库插入和提交操作。
    *   **优化**：将 `commit()` 移出实体处理循环，在整个 batch 处理完毕后统一提交一次。

---

## 2. 训练效率：TransE (`scripts/transe/10_train_transe.py`)

训练脚本的逻辑正确，但在 GPU 利用率和显存操作上有优化空间。

*   **问题 A：低效的 Anchor Locking**
    *   **现状**：为了保证锚点实体不更新，当前做法是在每个 batch 后使用 `index_copy_` 将初始向量复制回模型权重。这导致了频繁的 H2D / D2D 显存拷贝。
    *   **优化**：使用梯度屏蔽（Gradient Masking）。在 `opt.step()` 之前将锚点对应索引的梯度置为 0 (`grad[anchor_idx] = 0`)。这样无需并在显存中反复拷贝数据。

*   **问题 B：数据加载瓶颈**
    *   **现状**：`DataLoader` 运行在主进程 (`num_workers=0`)。
    *   **优化**：设置 `num_workers=4` 并开启 `pin_memory=True`，让 CPU 并行预取数据，避免 GPU 等待数据。

*   **问题 C：精度冗余**
    *   **现状**：使用 FP32 全精度训练。
    *   **优化**：集成 `torch.cuda.amp` 进行混合精度训练 (FP16)。这在现代 GPU 上能显著加速计算并减少显存占用。

---

## 3. 内存管理 (Low Priority)

### NER 推理 (`scripts/steps/01_ner_titles_nbbert.py`)

*   **问题**：启动时通过 `read_news_ids_and_titles` 将整个数据集（News ID 和 Title）一次性加载到内存列表。
*   **风险**：对于全量数据集，可能导致 OOM（内存溢出）。
*   **优化**：修改数据读取逻辑为生成器（Generator），流式读取和处理数据。

---

## 4. 推荐实施路线

建议按以下步骤逐步实施优化：

1.  **Immediate Fix (修复卡顿)**：优先修改 `scripts/steps/02_link_to_wikidata.py` 和 `src/adressa_entity/wikidata.py`，解决串行请求和数据库频繁提交的问题。
2.  **Stability Fix (稳定抓取)**：优化 `scripts/transe/09_fetch_wikidata_triples.py` 的数据库事务粒度。
3.  **Speed Up (加速训练)**：在 `scripts/transe/10_train_transe.py` 中实施梯度屏蔽和 DataLoader 优化。
