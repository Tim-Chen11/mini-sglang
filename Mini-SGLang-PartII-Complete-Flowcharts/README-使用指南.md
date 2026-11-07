# Mini-SGLang Part II 全链路流程图｜使用说明

核心成果是 `Mini-SGLang-PartII-Interactive.html`：**独立的单个 HTML 文件，不需要网络、无需安装任何前端包**，可以放大、拖拽、按模块切换，点击节点可打开相关教程。需要联网的是点击后进入 GitHub 的源码链接。

## 文件导航
- `01_master.svg` / `.png`：主生命周期，所有 10 章，包含反馈回路、流式返回和取消
- `02_http_process.svg`：启动进程、API、Tokenizer、Detokenizer、uid 并发隔离、Abort
- `03_scheduler.svg`：Scheduler 迭代、prefill-first、资源门槛、Chunked Prefill、Decode
- `04_kv_radix.svg`：KV Pool、Page Table、Radix Tree 3 者的区别与联系
- `05_forward_tp.svg`：模型 Forward + KV + Tensor Parallel
- `06_speculative.svg`：投机解码草稿、并行验证、Greedy vs Random、拒绝校正
- `07_batch_example.svg`：示意 A/B/C 不同生命周期怎样动态进入或离开计算 batch
- 对应 `.dot` 可用 [Graphviz](https://graphviz.org/) 编辑，或通过 `dot -Tsvg 01_master.dot -o my.svg` 重绘。

## 10 章对应节点
| 章 | 知识点 | 图中相关位置 |
|---|---|---|
| Ch1 | 逻辑分层 / Req / Batch / Context / 进程 | 主图 ①③⑤；详图 A |
| Ch2 | TokenizeMsg→UserMsg→DetokenizeMsg→UserReply | 主图 ②③⑦；详图 A |
| Ch3 | Embedding→Attention→MLP→LM Head→Sampler | 主图 ⑤；详图 D |
| Ch4 | KV 逐层写入 / 历史复用 | 主图 ④⑤；详图 C、D |
| Ch5 | async HTTP、uid、SSE、断连、Abort | 主图 ②⑦⑧；详图 A |
| Ch6 | 迭代调度、Prefill/Decode、资源预算、分块、动态接入 | 主图 ③⑥；详图 B、示例图 |
| Ch7 | page / slot / page_table / 分页读写 | 主图 ④；详图 C |
| Ch8 | 最长前缀、handle 锁、Radix Tree、缓存淘汰与保留 | 主图 ④⑥；详图 C |
| Ch9 | rank0 广播、各 rank 同步、Column/Row TP、All-Reduce | 主图 ①③⑤；详图 A、D |
| Ch10 | Draft→Verify→接受/拒绝/Bonus + KV 修正 | 主图 ⑥（可选）；详图 E |

## 四个重要的源码准确性细节

1. 第 6 章固定参考版是 **prefill-first**：`prefill_manager.schedule_next_batch(...) or decode_manager.schedule_next_batch()`。Continuous Batching 意味每轮动态决定参与计算的请求，并不要求把 prefill + decode 的请求在同一轮混算。
2. 官方 mini-sglang `page_table` 是每请求一行的 **token position → physical token slot**；Attention Backend 再转换成底层 kernel 使用的物理 page ID 布局。并非每个 Page Table 表项都是 page ID。
3. 有缓存匹配时，Scheduler 使用 `match_req` 取得 handle，检查容量、lock handle，确保实际使用阶段不会被 Radix eviction 收回。未使用的完整可复用前缀可能在请求结束后留在 Pool；资源释放 ≠ 立刻清空所有 KV。
4. 第 10 章的教学代码引用 **正式 SGLang** 的投机解码源文件，**不是** mini-sglang `9a91cfa` 基线里的现成实现。图将其标为可选扩展分支，而不是声称所有请求都经过此分支。

## 为什么首次 token 已经输出，但还没有它自己的 KV？

Prefill 计算 prompt `P`，产生 `P` 最后位置的 logits，采样获得 `y1`。此时 KV Pool 中已经有 `P` 的 K/V，但 `y1` 是刚刚采样出的下一个 token，尚未送进 Transformer。下一次 Decode 输入 `y1`，先写它的 K/V，再从新 logits 采样 `y2`。这个“差一拍”是理解 `input_ids`、`cached_len`、`extend_len` 的关键。

## 重要术语与消息

- `Req`：独立用户请求，包括输入、输出、已缓存长度、采样参数、cache handle
- `Batch`：当前模型一步要处理的请求集合和展开后的输入位置；`phase=prefill/decode`
- `Context`：本次执行的 AttentionBackend、KV Pool、Batch 与必要运行时状态
- `pending_list`：等待 prefill 或下一 chunk 的请求；`running_reqs`：可继续 decode 的请求
- `KV Cache Pool`：每层 K/V 的实际 GPU 张量；`free_slots`：物理页空闲表；`out_loc`：本轮新 K/V 的物理写入位置
- `page_table[table_idx, position]`：某个请求的逻辑 token 在实际显存中的物理 slot
- `Radix Tree`：将重复的 token 前缀指向已存在的 KV slots；不是存 K/V 数值的地方
- `ack_map`：API Server 内为每个 uid 缓存回复；`event_map`：唤醒该 uid 的异步消费者

## 主要核对来源

- [第 1 章：第1章_mini-sglang：推理引擎长什么样.md](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/%E7%AC%AC1%E7%AB%A0_mini-sglang%EF%BC%9A%E6%8E%A8%E7%90%86%E5%BC%95%E6%93%8E%E9%95%BF%E4%BB%80%E4%B9%88%E6%A0%B7.md)
- [第 2 章：第2章_一个请求的旅程_代码.md](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/%E7%AC%AC2%E7%AB%A0_%E4%B8%80%E4%B8%AA%E8%AF%B7%E6%B1%82%E7%9A%84%E6%97%85%E7%A8%8B_%E4%BB%A3%E7%A0%81.md)
- [第 3 章：第3章_前向与生成_代码.md](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/%E7%AC%AC3%E7%AB%A0_%E5%89%8D%E5%90%91%E4%B8%8E%E7%94%9F%E6%88%90_%E4%BB%A3%E7%A0%81.md)
- [第 4 章：第4章_KVCache优化_代码.md](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/%E7%AC%AC4%E7%AB%A0_KVCache%E4%BC%98%E5%8C%96_%E4%BB%A3%E7%A0%81.md)
- [第 5 章：第5章_HTTP服务与并发_代码.md](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/%E7%AC%AC5%E7%AB%A0_HTTP%E6%9C%8D%E5%8A%A1%E4%B8%8E%E5%B9%B6%E5%8F%91_%E4%BB%A3%E7%A0%81.md)
- [第 6 章：第6章_ContinuousBatching与调度_代码.md](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/%E7%AC%AC6%E7%AB%A0_ContinuousBatching%E4%B8%8E%E8%B0%83%E5%BA%A6_%E4%BB%A3%E7%A0%81.md)
- [第 7 章：第7章_PagedKVCache与显存管理.md](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/%E7%AC%AC7%E7%AB%A0_PagedKVCache%E4%B8%8E%E6%98%BE%E5%AD%98%E7%AE%A1%E7%90%86.md)
- [第 8 章：第8章_RadixAttention与前缀缓存_代码.md](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/%E7%AC%AC8%E7%AB%A0_RadixAttention%E4%B8%8E%E5%89%8D%E7%BC%80%E7%BC%93%E5%AD%98_%E4%BB%A3%E7%A0%81.md)
- [第 9 章：第9章_多进程与张量并行_代码.md](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/%E7%AC%AC9%E7%AB%A0_%E5%A4%9A%E8%BF%9B%E7%A8%8B%E4%B8%8E%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C_%E4%BB%A3%E7%A0%81.md)
- [第 10 章：第10章_投机解码_代码.md](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/%E7%AC%AC10%E7%AB%A0_%E6%8A%95%E6%9C%BA%E8%A7%A3%E7%A0%81_%E4%BB%A3%E7%A0%81.md)

- [mini-sglang 源码](https://github.com/sgl-project/mini-sglang)
- [正式 SGLang Speculative Decoding 参考](https://github.com/sgl-project/sglang/tree/1291f3175d3b7614821884e9653985fcd3c3dae8/python/sglang/srt/speculative)
