from graphviz import Digraph
from pathlib import Path
from urllib.parse import quote
import html,json,re,subprocess,os
from contextlib import contextmanager
OUT=Path('/mnt/data/minisglang_flowcharts')
BASE='https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/'
CH={
1:'第1章_mini-sglang：推理引擎长什么样.md',
2:'第2章_一个请求的旅程_代码.md',
3:'第3章_前向与生成_代码.md',
4:'第4章_KVCache优化_代码.md',
5:'第5章_HTTP服务与并发_代码.md',
6:'第6章_ContinuousBatching与调度_代码.md',
7:'第7章_PagedKVCache与显存管理.md',
8:'第8章_RadixAttention与前缀缓存_代码.md',
9:'第9章_多进程与张量并行_代码.md',
10:'第10章_投机解码_代码.md',
}
C={
'io':('#E6F3FF','#19669A'),
'sched':('#F0EAFF','#6C44B1'),
'gpu':('#E6F8EF','#13855F'),
'kv':('#FFF0DB','#A65D13'),
'parallel':('#FCE7F3','#AD4683'),
'spec':('#DCF6F2','#00877B'),
'logic':('#FFF6D5','#A87910'),
'error':('#FFEAE9','#B83333'),
'gray':('#EDF2F7','#64748B'),
}
font='Noto Sans CJK SC'

def graph(name, title, direction='LR', fontsize=17, nodesep='0.28', ranksep='0.55'):
 d=Digraph(name=name,format='svg',engine='dot')
 d.attr(rankdir=direction,compound='true',splines='spline',nodesep=nodesep,ranksep=ranksep,pad='0.32', bgcolor='#F9FBFF', outputorder='edgesfirst',fontname=font,label=title, labelloc='t',fontsize='26',fontcolor='#132238',fontweight='bold')
 d.attr('node',shape='box',style='rounded,filled',fontname=font,fontsize=str(fontsize),margin='0.18,0.13',penwidth='1.45',color='#64748B',fontcolor='#1E293B')
 d.attr('edge',color='#94A3B8',arrowsize='0.78',penwidth='1.6',fontname=font,fontsize='12',fontcolor='#475569')
 return d

@contextmanager
def sub(d,name,label,kind='gray'):
 with d.subgraph(name=('group_'+name if d.name=='master' and name=='scheduler' else 'cluster_'+name)) as ctx:
  ctx.attr(label='  '+label+'  ',fontname=font,fontsize='19',fontcolor=C[kind][1],color=C[kind][1],fillcolor='#FFFFFF',style='rounded,filled',penwidth='1.5',margin='20',labelloc='t',labeljust='l')
  yield ctx

def n(d,ID,s,kind='gray',ch=1,tip=None,shape='box'):
 f,b=C[kind]
 d.node(ID,label=s,shape=shape,fillcolor=f,color=b,URL=BASE+quote(CH[ch]),target='_blank',tooltip=(tip or s.replace('\n',' / '))+' | 第'+str(ch)+'章')

def e(d,a,b,label='',kind=None,**kwargs):
 p=kwargs.pop('penwidth',None)
 d.edge(a,b,label=label,color=(C.get(kind,('', '#97A6BC'))[1] if kind else '#8293A9'),penwidth=str(p or 1.55),**kwargs)

def legend(d):
 pass

def save(d,key,dpi=125):
 d.save(OUT/(key+'.dot'))
 svg_path=Path(d.render(filename=key, directory=str(OUT),format='svg',cleanup=False,quiet=True))
 # Compress SVG? no to keep vector editing possible
 # PNG from graphviz has better CJK font support than CairoSVG in this env
 subprocess.run(['dot',f'-Gdpi={dpi}','-Tpng',str(OUT/(key+'.dot')),'-o',str(OUT/(key+'.png'))],check=True)
 x=svg_path.read_text(); size=re.search(r'<svg[^>]*viewBox="([^"]+)"',x)
 print(key,'SVG',len(x),'png KB',round((OUT/(key+'.png')).stat().st_size/1024),'viewBox',size.group(1) if size else '?',flush=True)
 return x

# =================== 01. MASTER E2E ==========================
d=graph('master','总图 · 一个请求穿过 Mini-SGLang 的完整生命周期（第 1–10 章）','LR',15,'0.25','0.45')
with sub(d,'startup','① 启动与进程 [Ch1 / Ch9]','parallel') as q:
 n(q,'boot','python -m minisgl\n模型、TP size、KV 配置','parallel',9)
 n(q,'spawn','spawn 子进程\nScheduler/Engine × TP ranks\nTokenizer × N + Detokenizer','parallel',9)
 n(q,'weights','GPU 上加载权重分片\n初始化 Gloo / PyNCCL\n准备 KV Pool / Page Table','parallel',9)
 e(q,'boot','spawn'); e(q,'spawn','weights')
with sub(d,'frontend','② HTTP / Tokenizer [Ch2 / Ch5]','io') as q:
 n(q,'client','用户 POST\n/v1/chat/completions\nmessages / params / stream','io',5)
 n(q,'server','FastAPI + Pydantic 校验\n选择 Chat Template / prompt\n生成唯一 uid','io',5)
 n(q,'front_map','FrontendManager\nack_map[uid] / event_map[uid]\n建立异步请求状态','io',5)
 n(q,'tmsg','TokenizeMsg\nuid + 文本 + 采样参数','io',2)
 n(q,'token','Tokenizer Worker\nChat Template → input_ids','io',2)
 n(q,'umsg','UserMsg / BatchBackendMsg\nuid + IDs + SamplingParams','io',2)
 e(q,'client','server');e(q,'server','front_map');e(q,'front_map','tmsg');e(q,'tmsg','token');e(q,'token','umsg')
with sub(d,'scheduler','③ Scheduler 持续迭代 [Ch2 / Ch6 / Ch9]','sched') as q:
 n(q,'scheduler_header','③ Scheduler 核心：调度 / 排队 / Batch','sched',6)
 e(q,'scheduler_header','rank0',style='invis')
 n(q,'rank0','TP rank0 接收消息\n同步广播至其他 ranks\n所有 rank 同步请求状态','parallel',9)
 n(q,'state','_process_one_msg\n校验长度；登记 WAITING\nPrefillManager.pending_list','sched',6)
 n(q,'loop','Scheduler.normal_loop / overlap_loop\n收消息 → 组批 → 执行 → 处理结果','sched',6)
 n(q,'choice','_schedule_next_batch\n能安排 Prefill 吗？','logic',6,shape='diamond')
 n(q,'pf','PrefillAdder\n新请求 + ChunkedReq\n匹配前缀 / 检查预算','sched',6)
 n(q,'dc','DecodeManager\n运行请求可继续？\n动态挑选当前 batch','sched',6)
 n(q,'no_work','无可运行 batch\n等待新消息或资源','gray',6)
 n(q,'make_batch','_prepare_batch\nBatch(phase, reqs)\ninput_ids / positions / out_loc','sched',6)
 e(q,'rank0','state');e(q,'state','loop');e(q,'loop','choice');e(q,'choice','pf','可',kind='sched');e(q,'choice','dc','不可',kind='sched');e(q,'pf','make_batch','组出 prefill batch');e(q,'dc','make_batch','组出 decode batch');e(q,'dc','no_work','没有',constraint='false')
 e(q,'no_work','loop',style='dashed',constraint='false')
with sub(d,'memory','④ Prefill 复用 / KV 内存 [Ch4 / Ch7 / Ch8]','kv') as q:
 n(q,'prefix','RadixTree match_prefix\n最长相同 token 前缀\n输出 cached_len / handle','kv',8)
 n(q,'mem_gate','槽位 / prefill token 预算\nKV estimated_len + reserved_size\n是否足够？','logic',6,shape='diamond')
 n(q,'evict','空闲页不足：尝试逐出\n未锁定的可淘汰前缀\n否则本轮无法接入','kv',8)
 n(q,'lock','Lock 命中前缀\n分配 request table row\n复制已缓存 slot 映射','kv',8)
 n(q,'alloc','CacheManager.allocate_paged\n为未缓存 token 分配物理页\n更新 page_table / out_loc','kv',7)
 n(q,'meta','AttentionBackend.prepare_metadata\n逻辑位置 → 物理 page/slot\ncache_seqlens / causal mask','kv',4)
 e(q,'prefix','mem_gate');e(q,'mem_gate','evict','不足',kind='kv');e(q,'evict','mem_gate','再检查',style='dashed',constraint='false');e(q,'mem_gate','lock','足够',kind='kv');e(q,'lock','alloc');e(q,'alloc','meta')
with sub(d,'forward','⑤ GPU Forward / TP [Ch3 / Ch4 / Ch9]','gpu') as q:
 n(q,'enter','Engine.forward_batch\nContext.batch ← 当前 Batch\nPrefill: 多个未缓存 token\nDecode: 每请求通常 1 token','gpu',3)
 n(q,'embed','Token Embedding + position\n各 Transformer Decoder Layers','gpu',3)
 n(q,'qkv','RMSNorm + QKV Projection\nQK Norm / RoPE / GQA\nTP: 切分 Head / 权重','gpu',9)
 n(q,'kvio','按 out_loc 写入本轮 K/V\n读 page_table 对应历史 KV\nCausal Attention','kv',4)
 n(q,'mlp','O Projection + Residual\nRMSNorm + Gated MLP\nTP Row Parallel → All-Reduce','gpu',9)
 n(q,'logits','Final Norm + LM Head\n每请求最后位置 logits\n→ Sampler','gpu',3)
 e(q,'enter','embed');e(q,'embed','qkv');e(q,'qkv','kvio');e(q,'kvio','mlp');e(q,'mlp','logits')
with sub(d,'result','⑥ 生成 / 更新 / 复用 [Ch6 / Ch8 / Ch10]','spec') as q:
 n(q,'spec_gate','仅 Decode 可选的扩展分支\nSpeculative 替代常规 Forward+Sample\nmini-sglang 基线不包含','logic',10,shape='diamond')
 n(q,'normal_sample','Normal sampler\nTemperature / Top-k / Top-p\n生成 1 个下一 token','gpu',3)
 n(q,'spec_draft','Draft k 个候选\nTarget 一次验证\n连续接受 + 纠正/bonus','spec',10)
 n(q,'prefill_chunk','本轮是未完成的 Chunked Prefill？\n若是，仅写 KV；不采样输出','logic',6,shape='diamond')
 n(q,'write','Req.append_host(output token)\n正式提交生成的 token(s)\n更新长度与请求状态','sched',6)
 n(q,'stop','达到 max_tokens?\nEOS 且非 ignore_eos？','logic',6,shape='diamond')
 n(q,'keep','未结束：留在 running 集合\n下一轮 Decode / 接收新请求','sched',6)
 n(q,'recycle','已结束/Abort：\n移除 waiting/running\n释放 table row 与 KV handle','kv',8)
 n(q,'persist','cache_req / insert_prefix\n保留可复用完整前缀\n解锁 / 归还其余页面','kv',8)
 e(q,'spec_gate','spec_draft','是：扩展路径');e(q,'prefill_chunk','normal_sample','否：正常采样');e(q,'normal_sample','write');e(q,'spec_draft','write','验证通过后提交');e(q,'write','stop');e(q,'stop','keep','否');e(q,'stop','recycle','是');e(q,'recycle','persist')
with sub(d,'response','⑦ 逐 token 返回客户端 [Ch2 / Ch5]','io') as q:
 n(q,'dmsg','DetokenizeMsg\nuid + next_token + finished','io',2)
 n(q,'detok','Detokenizer Worker\n增量解码 UTF-8 / token IDs\nUserReply(uid, delta, finished)','io',5)
 n(q,'listener','API listen()\n按 uid 写 ack_map[uid]\n设置 event_map[uid] 唤醒','io',5)
 n(q,'mode','HTTP stream == true ?','logic',5,shape='diamond')
 n(q,'sse','StreamingResponse SSE\ndata: delta.content\n最终 data: [DONE]','io',5)
 n(q,'json','收集完整文本\n返回完整 JSON Response','io',5)
 n(q,'done','客户端看到增量文本\n或最终回答','io',5)
 e(q,'dmsg','detok');e(q,'detok','listener');e(q,'listener','mode');e(q,'mode','sse','是');e(q,'mode','json','否');e(q,'sse','done');e(q,'json','done')
with sub(d,'abort','⑧ 并发取消与清理 [Ch5 / Ch6]','error') as q:
 n(q,'disconnect','连接断开 / async cancel','error',5)
 n(q,'abmsg','API 清除 per-uid 状态\nAbortMsg → Tokenizer\nAbortBackendMsg → Scheduler','error',5)
 n(q,'abort_sched','按 uid 从 pending / running 移除\n已在 GPU 的工作可有迟到回复\n调用统一资源释放路径','error',6)
 e(q,'disconnect','abmsg');e(q,'abmsg','abort_sched')
# lane transitions
for a,b in [('weights','client'),('umsg','rank0'),('make_batch','alloc'),('meta','enter'),('logits','prefill_chunk'),('stop','dmsg')]:e(d,a,b,kind='gray',penwidth=2.15)
e(d,'pf','prefix',style='dashed',kind='kv',constraint='false')
e(d,'make_batch','spec_gate','可选投机旁路',style='dotted',kind='spec',constraint='false')
e(d,'mem_gate','pf','无法接入：留队列',style='dotted',constraint='false')
e(d,'prefill_chunk','loop','是：下轮继续 prefill',kind='sched',style='dashed',constraint='false')
e(d,'keep','loop','下一轮继续',kind='sched',style='dashed',constraint='false')
e(d,'persist','loop','资源腾出，继续调度',kind='kv',style='dotted',constraint='false')
e(d,'client','disconnect','用户中途离开',kind='error',style='dashed',constraint='false')
e(d,'abort_sched','recycle',kind='error',style='dashed',constraint='false')
e(d,'recycle','dmsg','若需发送最终状态',style='dotted',constraint='false')
svg_master=save(d,'01_master',113)

# =================== 02. HTTP + PROCESS ==========================
d=graph('http','详图 A · 启动、多进程、HTTP 并发、Streaming 与 Abort（Ch1 / 2 / 5 / 9）','LR',17,'0.30','0.56')
with sub(d,'init','进程启动：server/launch.py [Ch1 / 9]','parallel') as q:
 for a,b,t in [('a0','CLI 参数','python -m minisgl；model/tp_size/page_size'),('a1','start_backend()','mp.spawn / ack_queue'),('a2','rank N','Scheduler + Engine / GPU N'),('a3','Tokenizer Workers','num_tokenizers 个分词进程'),('a4','Detokenizer','1 个反分词进程'),('a5','Front API','uvicorn / FastAPI')]:
  n(q,a,b+'\n'+t,'parallel',9,shape='box')
 e(q,'a0','a1');e(q,'a1','a2');e(q,'a1','a3');e(q,'a1','a4');e(q,'a1','a5')
with sub(d,'front','HTTP API Server + FrontendManager','io') as q:
 n(q,'b0','Client POST JSON\n/messages prompt, max_tokens, stream','io',5)
 n(q,'b1','Pydantic Validation\nmodel + sampling 参数\n提取 / 拼接 prompt','io',5)
 n(q,'b2','new_user(): uid\nack_map[uid] = []\nevent_map[uid] = Event()','io',5)
 n(q,'b3','send_one(TokenizeMsg)\nZMQ Push Queue','io',5)
 n(q,'b4','listen() 常驻协程\nZMQ 收 UserReply','io',5)
 n(q,'b5','按 uid 路由\nack_map[uid].append(reply)\nevent_map[uid].set()','io',5)
 n(q,'b6','wait_for_ack(uid)\nawait Event.wait → clear\n取出未消费 UserReply','io',5)
 n(q,'b7','stream?','logic',5,shape='diamond')
 n(q,'b8','SSE: text_completion.chunk\n增量 delta.content\n结束发送 [DONE]','io',5)
 n(q,'b9','Non-stream: 累积增量\n等待 finished → JSON','io',5)
 n(q,'b10','Client 接收回答','io',5)
 for x,y in [('b0','b1'),('b1','b2'),('b2','b3'),('b4','b5'),('b5','b6'),('b6','b7'),('b7','b8'),('b7','b9'),('b8','b10'),('b9','b10')]:e(q,x,y)
with sub(d,'backend','Tokenizer → Scheduler → Engine → Detokenizer','sched') as q:
 n(q,'c0','Tokenizer Worker\nChat Template → encode → input_ids','io',2)
 n(q,'c1','UserMsg(uid, IDs, params)\nBatchBackendMsg','io',2)
 n(q,'c2','rank0 Scheduler 收消息\nTP>1: Gloo 广播消息数\nZMQ Pub 实际消息','parallel',9)
 n(q,'c3','所有 TP rank 同序处理\nPrefill/Decode + Model Forward\nSampler 输出 token IDs','sched',6)
 n(q,'c4','rank0 → DetokenizeMsg\nuid / token ID / finished','io',2)
 n(q,'c5','Detokenizer Worker\n增量解码（处理 token 边界）\nUserReply(uid, delta, finished)','io',2)
 for x,y in [('c0','c1'),('c1','c2'),('c2','c3'),('c3','c4'),('c4','c5')]:e(q,x,y)
with sub(d,'abortion','取消/断线分支 — 协作式，并非强杀 GPU','error') as q:
 n(q,'x0','stream 客户端断开\n或 CancelledError','error',5)
 n(q,'x1','API abort_user(uid)\n删前台状态 → AbortMsg','error',5)
 n(q,'x2','Tokenizer 转换\nAbortBackendMsg','error',5)
 n(q,'x3','Scheduler 按 uid 取消\npending / running 内删除\n释放页表行与缓存句柄','error',6)
 n(q,'x4','可能存在 GPU 在途计算\n迟到回复按 uid 忽略','error',5)
 for x,y in [('x0','x1'),('x1','x2'),('x2','x3'),('x3','x4')]:e(q,x,y)
for x,y in [('a5','b0'),('b3','c0'),('c5','b4'),('b8','x0'),('x1','x2'),('x2','x3'),('x3','c3')]:
 e(d,x,y,style='dashed' if x.startswith('x') else 'solid',constraint='false' if x.startswith('x') else 'true',kind='error' if x.startswith('x') else None)
e(d,'c3','c4','每轮产出 token',constraint='false')
svg_http=save(d,'02_http_process',120)

# =================== 03. SCHEDULER =============================
d=graph('sched','详图 B · Scheduler / Continuous Batching / Prefill / Decode（Ch6 + Ch7/8 资源约束）','TB',17,'0.28','0.47')
with sub(d,'states','请求状态与调度循环','sched') as q:
 n(q,'s0','BatchBackendMsg / AbortBackendMsg 到达\n_process_one_msg()','sched',6)
 n(q,'s1','UserMsg → 校验输入长度\nReq / PendingReq 加入 pending_list','sched',6)
 n(q,'s2','等待区 WAITING\npending_list: 新请求 + ChunkedReq','sched',6)
 n(q,'s3','运行区 RUNNING\nDecodeManager.running_reqs\n每请求独立 token / KV 状态','sched',6)
 n(q,'s4','normal_loop 迭代\nreceive_msg(blocking=idle)\n_schedule_next_batch()','sched',6)
 n(q,'s5','优先尝试 Prefill\nPrefillManager.schedule_next_batch(budget)','sched',6)
 n(q,'s6','Prefill 是否可安排？','logic',6,shape='diamond')
 n(q,'s7','否则 DecodeManager\nschedule_next_batch()','sched',6)
 n(q,'s8','Batch 存在？','logic',6,shape='diamond')
 n(q,'s9','等待外部请求 / 可释放空间','gray',6)
 e(q,'s0','s1',label='UserMsg');e(q,'s1','s2');e(q,'s2','s4');e(q,'s3','s4',constraint='false');e(q,'s4','s5');e(q,'s5','s6');e(q,'s6','s7','否');e(q,'s7','s8');e(q,'s8','s9','否');e(q,'s9','s4',constraint='false',style='dashed')
with sub(d,'pref','PrefillAdder：动态接入、资源校验、Chunked Prefill','kv') as q:
 n(q,'p0','遍历 pending_list（按顺序）\n若本轮无法接入 → break','sched',6)
 n(q,'p1','已有 chunked_req ?','logic',6,shape='diamond')
 n(q,'p2','新请求：CacheManager.match_req\n匹配 input_ids[:-1]\ncached_len / handle','kv',8)
 n(q,'p3','约束 ① request row 空闲？\nTableManager.available_size > 0','logic',6,shape='diamond')
 n(q,'p4','约束 ② KV 容量足够？\nextend_len = input_len - cached_len\nestimated_len = extend_len + output_len\n+ reserved_size ≤ available_size','logic',6,shape='diamond')
 n(q,'p5','锁定前缀 handle.lock\n再次检查容量 + 分配 table_idx','kv',8)
 n(q,'p6','约束 ③ prefill token_budget\nmin(remaining_prompt, token_budget)','logic',6,shape='diamond')
 n(q,'p7','本轮 Prefill 完整\n创建普通 Req','sched',6)
 n(q,'p8','本轮只计算部分 prompt\n创建 ChunkedReq\n未完成部分回 pending 队首','sched',6)
 n(q,'p9','资源不足 / 预算已用尽\n无法接入 → 等后续迭代','error',6)
 n(q,'p10','Batch(reqs, phase=prefill)\n准备 prefill 计算','sched',6)
 for x,y in [('p0','p1'),('p1','p2'),('p2','p3'),('p3','p4'),('p4','p5'),('p5','p6'),('p6','p7'),('p6','p8'),('p7','p10'),('p8','p10')]:e(q,x,y)
 e(q,'p3','p9','否',constraint='false');e(q,'p4','p9','否',constraint='false');e(q,'p1','p6','已有 chunked',style='dashed',constraint='false')
with sub(d,'dec','Decode / 动态 Batch / 每轮处理','gpu') as q:
 n(q,'d0','filter_reqs：运行集合\n丢弃 finished / aborted','sched',6)
 n(q,'d1','保证 Decode KV 页容量\n运行中请求可再处理？','logic',6,shape='diamond')
 n(q,'d2','Batch(reqs, phase=decode)\n每请求本轮通常输入 1 token','sched',6)
 n(q,'d3','_prepare_batch()\n分配本轮 KV slots / positions\nAttention metadata','kv',7)
 n(q,'d4','_forward(Batch) → Engine.forward_batch\n一个 batch 完成一次 GPU step','gpu',3)
 n(q,'d5','_process_last_data\nGPU token → CPU\n逐 Req 更新 input_ids/len','sched',6)
 n(q,'d6','ChunkedReq?','logic',6,shape='diamond')
 n(q,'d7','不发 output token\n继续等剩余 prefill chunk','sched',6)
 n(q,'d8','EOS / max_tokens / Abort?','logic',6,shape='diamond')
 n(q,'d9','未完成：加入 / 保留 DecodeManager\n每轮可以有 A、B 加入与离开','sched',6)
 n(q,'d10','完成：remove_req / table.free\ncache_req(finished=True)\n回收/归档 KV','kv',8)
 n(q,'d11','DetokenizeMsg\nuid + 新 token + finished','io',5)
 for x,y in [('d0','d1'),('d1','d2'),('d2','d3'),('d3','d4'),('d4','d5'),('d5','d6'),('d6','d8'),('d6','d7'),('d8','d9'),('d8','d10'),('d8','d11')]:e(q,x,y)
 e(q,'d1','d0','暂缓',constraint='false',style='dotted');e(q,'d9','d0','下一迭代',constraint='false',style='dashed')
e(d,'s6','p0','是',kind='sched');e(d,'p10','d3',kind='sched');e(d,'s8','d0','是',kind='sched');e(d,'p9','s7','无法排入 Prefill',kind='error',style='dashed',constraint='false');e(d,'d7','s2','下轮 chunk',style='dashed',constraint='false');e(d,'d9','s3',style='dotted',constraint='false');e(d,'d10','s4','释放资源后继续',style='dotted',constraint='false')
svg_sched=save(d,'03_scheduler',120)

# ==================== 04. KV / RADIX =============================
d=graph('kv','详图 C · KV Cache 的 3 种对象：Pool ≠ Page Table ≠ Radix Tree（Ch4 / 7 / 8）','LR',17,'0.32','0.56')
with sub(d,'reference','新请求：前缀命中 / 分配','kv') as q:
 n(q,'k0','input_ids = [A,B,C,D,E]\n本请求已有哪些 token？','io',4)
 n(q,'k1','RadixTree longest-prefix match\n命中已存的 [A,B,C]\n匹配按 page_size 对齐','kv',8)
 n(q,'k2','CacheHandle(cached_len,node)\n收集命中节点的物理 slot indices','kv',8)
 n(q,'k3','锁住匹配的前缀节点\n避免正在使用时被 LRU 淘汰','kv',8)
 n(q,'k4','TableManager.allocate\n分配当前 Req 的 page_table row','kv',7)
 n(q,'k5','复制命中段的 slot indices\n未命中段需要新 page','kv',4)
 e(q,'k0','k1');e(q,'k1','k2');e(q,'k2','k3');e(q,'k3','k4');e(q,'k4','k5')
with sub(d,'phys','物理内存：Pool + Free Pages','kv') as q:
 n(q,'f0','KV Cache Pool（真实 GPU 张量）\n每层 K_buffer / V_buffer\n[pages, page_size, local_kv_heads, dim]','kv',4)
 n(q,'f1','free_slots：空闲物理页起始 slot\n例：page_size=4, [0,4,8,12]','kv',7)
 n(q,'f2','KV 页容量不够？','logic',7,shape='diamond')
 n(q,'f3','RadixCache.evict()\n逐出未锁定的 LRU 叶子前缀\n其物理页重新进入 free_slots','kv',8)
 n(q,'f4','allocate_paged → 物理页\n把每页展开为 token slot\n写到 batch.out_loc','kv',7)
 n(q,'f5','一个 layer 的 store_kv\n(indices = batch.out_loc)\n写入本轮新 K/V','kv',4)
 e(q,'f1','f2');e(q,'f2','f3','是');e(q,'f3','f4');e(q,'f2','f4','否');e(q,'f4','f5');e(q,'f5','f0')
with sub(d,'mapping','Page Table：逻辑位置 → 物理 token slot','kv') as q:
 n(q,'t0','page_table[table_idx, position]\n存的是物理 token slot\n不是原始 page ID','kv',4)
 n(q,'t1','实例（page_size=4）\n请求逻辑位置 [0 1 2 3 4]\n物理 slots [8 9 10 11 20]','kv',7)
 n(q,'t2','Attention Backend.prepare_metadata\n转换为 kernel 需要的物理 Page IDs\n+ cache_seqlens + causal info','kv',4)
 n(q,'t3','Attention: Q 当前 token\n按 page_table 读取可见历史 K/V\n在线 Softmax / FlashAttention','gpu',4)
 e(q,'t0','t1');e(q,'t1','t2');e(q,'t2','t3')
with sub(d,'return','结果完成 → 前缀插入 / 回收','kv') as q:
 n(q,'r0','Prefill 完成非 chunk\ncache_req(finished=False)\n登记已计算前缀，保留 handle','kv',8)
 n(q,'r1','Decode 若生成 1 token\n扩展 request page_table\n本轮不会每次插入 RadixTree','kv',8)
 n(q,'r2','Finished / Cancelled\n_free_req_resources()\nTableManager.free(table_idx)','kv',8)
 n(q,'r3','cache_req(finished=True)\ninsert_prefix + unlock(old_handle)\n可复用完整页保留在树中','kv',8)
 n(q,'r4','多余 / 重复 / 非整页尾部\n归还 free_slots（无需把显存清零）','kv',8)
 n(q,'r5','未来用户命中同样前缀\n避免重新计算其 K/V\n降低 Prefill TTFT','gpu',8)
 for x,y in [('r0','r1'),('r1','r2'),('r2','r3'),('r3','r4'),('r3','r5')]:e(q,x,y)
for x,y in [('k5','f1'),('k5','t0'),('f4','t0'),('f0','t3'),('t3','r0'),('r4','f1'),('r5','k1')]:e(d,x,y,style='dashed' if x in ['r4','r5'] else 'solid',constraint='false' if x in ['r4','r5'] else 'true')
svg_kv=save(d,'04_kv_radix',118)

# ==================== 05. FORWARD + TP ==========================
d=graph('forward','详图 D · Qwen3 Forward、KV 读写与 Tensor Parallel（Ch3 / 4 / 9）','LR',17,'0.30','0.53')
with sub(d,'part','进程 / 权重 / 数据切分（TP size = N）','parallel') as q:
 n(q,'w0','同一 Batch：所有 TP rank\n接收同序消息 / 本地状态一致','parallel',9)
 n(q,'w1','每 rank 一个 Engine\nset_tp_info(rank,size)\n加载局部 weight shard','parallel',9)
 n(q,'w2','CPU 控制同步：Gloo\nGPU 集合通信：PyNCCL / NCCL\n单卡则无跨卡通信','parallel',9)
 e(q,'w0','w1');e(q,'w1','w2')
with sub(d,'model','一轮 Forward（Prefill / Decode 共用 Layer）','gpu') as q:
 n(q,'m0','Batch.input_ids + positions\nPrefill: extend_len>1\nDecode: 每 Req 通常 1','gpu',3)
 n(q,'m1','VocabParallelEmbedding\n查 token embedding → Hidden X','gpu',9)
 n(q,'m2','Fused RMSNorm + Residual\n每层第一处 norm','gpu',3)
 n(q,'m3','QKV Projection\nColumn Parallel / 按 head 切分','parallel',9)
 n(q,'m4','Q/K RMSNorm + RoPE\nGQA: Hq > Hkv\n不同 Q 头共享 KV heads','gpu',3)
 n(q,'m5','KV Cache.store_kv(K,V,out_loc)\n本 layer 将新 token K/V 写入显存','kv',4)
 n(q,'m6','Causal Attention\n读取历史 KV（Page Table）\n只看当前及以前 token','gpu',4)
 n(q,'m7','O Projection（Row Parallel）\nAll-Reduce 合并 TP 局部结果\n+ Residual','parallel',9)
 n(q,'m8','RMSNorm → GatedMLP\nGate/Up Column Parallel\nSiLU(gate) * up','gpu',3)
 n(q,'m9','Down Projection Row Parallel\nAll-Reduce + Residual','parallel',9)
 n(q,'m10','下一 Decoder Layer ?','logic',3,shape='diamond')
 n(q,'m11','Final RMSNorm\n每请求选最后预测位置','gpu',3)
 n(q,'m12','Parallel LM Head → logits [B,V]\n不同 rank 可参与词表采样','gpu',9)
 n(q,'m13','Sampler: greedy / temp / top-k / top-p\n选择下一 token id','gpu',3)
 for x,y in [('m0','m1'),('m1','m2'),('m2','m3'),('m3','m4'),('m4','m5'),('m5','m6'),('m6','m7'),('m7','m8'),('m8','m9'),('m9','m10'),('m10','m11'),('m11','m12'),('m12','m13')]:e(q,x,y)
 e(q,'m10','m2','还有 layers',constraint='false',style='dashed')
with sub(d,'cycle','自回归循环：生成 token 与 KV 状态错位一拍','sched') as q:
 n(q,'z0','Prefill 已计算 prompt 的 K/V\n首次 logits 预测 y1','sched',4)
 n(q,'z1','Sample y1 → 追加 token id\n注意 y1 的 K/V 尚未产生','sched',4)
 n(q,'z2','下轮 Decode 输入 y1\n算出 K(y1)/V(y1) 并写缓存\nlogits 再预测 y2','sched',4)
 n(q,'z3','不断迭代 → EOS / max_tokens\n或者返回到 Scheduler 再组批','sched',6)
 e(q,'z0','z1');e(q,'z1','z2');e(q,'z2','z3')
e(d,'w2','m0');e(d,'m13','z0');e(d,'z2','m0',style='dashed',constraint='false')
svg_forward=save(d,'05_forward_tp',116)

# ==================== 06. SPECULATIVE ==========================
d=graph('spec','详图 E · Speculative Decoding：Draft → Verify → Accept / Reject（第 10 章，SGLang 扩展）','TB',18,'0.31','0.48')
with sub(d,'premise','前置条件与状态','spec') as q:
 n(q,'t0','当前已确认前缀 P\nTarget 大模型已准备 KV\nDraft 小模型有独立状态','spec',10)
 n(q,'t1','两模型 token ID / 词表语义兼容\n本轮仍有 remaining token 配额','spec',10)
 n(q,'t2','Draft 自回归连续提出 k 个候选\n[y1, y2, ..., yk]\n候选不是正式输出！','spec',10)
 n(q,'t3','构造 VerifyInput\n候选 IDs / positions / masks / indices\n随机方案还需要 q_i 概率','spec',10)
 n(q,'t4','Target 一次并行验证候选\n因果 Attention 保证每行只看此前文\n得到 k 个位置预测 + 可能的 bonus 预测','gpu',10)
 e(q,'t0','t1');e(q,'t1','t2');e(q,'t2','t3');e(q,'t3','t4')
with sub(d,'choice','两条验证策略','logic') as q:
 n(q,'c0','目标模型原始解码策略？','logic',10,shape='diamond')
 n(q,'c1','Greedy 验证\n逐位置比较 draft token\n与 target argmax','spec',10)
 n(q,'c2','Stochastic 验证\np_i = Target 实际采样分布\nq_i = Draft 实际采样分布','spec',10)
 n(q,'c3','对候选 y：\nα=min(1, p(y)/q(y))\n概率 α 接受','spec',10)
 n(q,'c4','接受了多少个连续草稿？','logic',10,shape='diamond')
 e(q,'c0','c1','贪心');e(q,'c0','c2','随机');e(q,'c2','c3');e(q,'c1','c4');e(q,'c3','c4')
with sub(d,'commit','第一次拒绝处停止；只提交真实已确认 token','spec') as q:
 n(q,'q0','首次不匹配 / 拒绝\n只保留之前连续接受的 a 个候选','spec',10)
 n(q,'q1','Greedy：提交目标 argmax 纠正 token\n随机：从 normalized max(p-q,0) 抽样\n丢弃后续所有草稿','spec',10)
 n(q,'q2','全部 k 个被接受\n用 Target 最后一行采样\n再提交 1 个 bonus token','spec',10)
 n(q,'q3','本轮通常提交 a+1 个 token\n最多 k+1；必须截断到 remaining','spec',10)
 n(q,'q4','遇到正式确认的 EOS?\n或 max_tokens 额度耗尽?','logic',10,shape='diamond')
 n(q,'q5','接受路径 KV 与 Draft 状态同步\n丢弃拒绝候选对应的缓存\n新纠正/bonus token 的 KV 留待下轮','kv',10)
 n(q,'q6','停止生成 / 输出 finished','io',10)
 n(q,'q7','以新确认前缀重新 Draft\n进入下一轮推测','spec',10)
 e(q,'q0','q1');e(q,'q1','q3');e(q,'q2','q3');e(q,'q3','q4');e(q,'q4','q5','否');e(q,'q4','q6','是');e(q,'q5','q7');e(q,'q7','t2',constraint='false',style='dashed')
e(d,'t4','c0');e(d,'c4','q0','部分接受',kind='spec');e(d,'c4','q2','全部接受',kind='spec')
svg_spec=save(d,'06_speculative',121)

# ==================== 07. BATCH EXAMPLE ==========================
d=graph('batch','额外实例 · 三个请求 A/B/C：同一个 Scheduler 如何连续组批（示意，不是固定时刻表）','LR',18,'0.36','0.53')
with sub(d,'reqs','3 个请求：独立状态','io') as q:
 n(q,'ar','A：长 prompt\n需要 Chunked Prefill','io',6)
 n(q,'br','B：短 prompt\n可以一次 Prefill','io',6)
 n(q,'cr','C：与 A 命中共同前缀\nRadix 命中，extend_len 短','io',8)
with sub(d,'ticks','Scheduler 按轮重新组 Batch','sched') as q:
 n(q,'tick1','Round 1\nPrefill A (chunk 1)\n仅建立一部分 KV，无输出','sched',6)
 n(q,'tick2','Round 2\nPrefill A (chunk 2)\n可处理 B 的 Prefill（受预算约束）','sched',6)
 n(q,'tick3','Round 3\nA / B 完整 Prefill 结束\n产生首 token，进入 running','sched',6)
 n(q,'tick4','Round 4\nDecode A + B\nC 新到达进入 pending','sched',6)
 n(q,'tick5','Round 5\nPrefill C：复用共同前缀\n输入有效 token 数减少','sched',8)
 n(q,'tick6','Round 6\nDecode A + B + C\nB 如果完成，立即移出','sched',6)
 n(q,'tick7','Round 7+\n只剩 A / C 继续 Decode\n空闲 slots 可接纳新请求','sched',6)
 for x,y in [('tick1','tick2'),('tick2','tick3'),('tick3','tick4'),('tick4','tick5'),('tick5','tick6'),('tick6','tick7')]:e(q,x,y)
with sub(d,'implication','为什么说是 Continuous Batching','gpu') as q:
 n(q,'comp1','每一轮 batch 都可能不同\n无需等 B 结束才能服务 C','gpu',6)
 n(q,'comp2','动态选择 prefill / decode\n而不是固定 batch 永远绑死','gpu',6)
 n(q,'comp3','不同请求仍有独立 uid、KV 映射、停止条件\n固定模型权重在 batch 内共享','gpu',6)
 e(q,'comp1','comp2');e(q,'comp2','comp3')
for x,y in [('ar','tick1'),('br','tick2'),('cr','tick5'),('tick7','comp1')]:e(d,x,y)
svg_batch=save(d,'07_batch_example',116)

# write all svg to embed later in HTML
all_svgs=[svg_master,svg_http,svg_sched,svg_kv,svg_forward,svg_spec,svg_batch]
paths=['01_master','02_http_process','03_scheduler','04_kv_radix','05_forward_tp','06_speculative','07_batch_example']
# SVG contains XML declaration, DOCTYPE. extract content <svg ...>...</svg>
svgs=[]
for s in all_svgs:
 m=re.search(r'(<svg\b[\s\S]*</svg>)',s)
 if not m: raise ValueError('could not extract svg')
 svgs.append(m.group(1))

chapterlinks=[]
for num,fname in CH.items():
 chapterlinks.append(f'<a href="{BASE+quote(fname)}" target="_blank" rel="noopener noreferrer"><b>Ch{num}</b> {html.escape(fname.removesuffix("_代码.md").removesuffix(".md"))}</a>')

labels=[
('总图','01 · END-TO-END','从启动到 API、Scheduler、KV、Forward、输出、取消'),
('前后台','02 · HTTP & PROCESSES','uid 与消息传递、流式响应、并发和取消'),
('调度器','03 · SCHEDULER','Prefill / Decode 决策、预算、Chunked Prefill'),
('KV 内存','04 · KV CACHE','KV Pool、Page Table、Radix Tree、锁与回收'),
('模型执行','05 · MODEL & TP','Transformer Forward、Attention 与多卡通信'),
('投机解码','06 · SPECULATIVE','Draft、Target Verify、接受/拒绝与 KV 修正'),
('三请求示例','07 · DYNAMIC BATCH','A/B/C 动态组批的时间线示意'),
]
notes=[
('全链路总图','蓝：HTTP/消息 · 紫：调度 · 橙：显存 · 绿：GPU · 粉：TP · 青：Speculative。实线是主要数据路径，虚线通常表示反馈、条件或跨模块联系。查看时请从 Client POST 开始，一直跟到 Detokenizer → SSE。'),
('请求与进程','用户请求经 FastAPI → TokenizeMsg → Tokenizer → UserMsg → Scheduler；返回是 DetokenizeMsg → Detokenizer → UserReply → ack_map/event_map。客户端断开后是协作式 Abort，GPU 已提交的工作并不立即被强制杀掉。'),
('连续批处理','实现中的 `_schedule_next_batch` 优先尝试 prefill，否则 decode。PrefillAdder 检查 request slot、KV 容量和本轮 token_budget；长 prompt 可以 Chunked Prefill，中间 chunk 不采样、不进入 Decode。'),
('存储层：3 个结构','KV Pool 是 GPU 实际 K/V；page_table 每行对应一个活动 Req，表项存物理 token slot；Radix Tree 将 token 前缀映射到共享的 KV slot 范围。命中后要 lock；结束后可插入完整前缀并释放不可复用页。'),
('神经网络与 TP','每一层将新增 K/V 写入 pool，并用 page table 对历史 KV 做因果 Attention。Column-Parallel 对输出/head 方向切分；Row-Parallel 的局部输出需要通信合并。Prefill logits 产生第一个输出 token，但这个 token 自身的 K/V 要到下一轮 Decode 才计算。'),
('投机解码不是 mini 基线','第 10 章代码引用的是正式 SGLang 的固定版本，不要误以为 mini-sglang 原始基线自带该模块。先 Draft k 个 token，再 Target 一次 Verify，首次拒绝就停止、纠正，全部接受则可 Bonus；随机版要使用正残差修正，才能保持目标采样分布。'),
('A / B / C 是教学例子','这个标签展示请求独立进出 batch 的原则，并非源代码必然的逐轮调度序列。实际哪个 prefill 或 decode 在某轮运行由 budget、资源、排队顺序和配置共同决定。'),
]

# Embed raw SVGs in hidden templates; self-contained interactive viewer.
templates='\n'.join(f'<template id="diagram-{i}">{s}</template>' for i,s in enumerate(svgs))
buttons='\n'.join(f'<button class="nav-item {"active" if i==0 else ""}" data-index="{i}"><span class="nav-idx">{tag}</span><strong>{cn}</strong><small>{subtitle}</small></button>' for i,(cn,tag,subtitle) in enumerate(labels))
rightnotes='\n'.join(f'<section class="help-entry" data-info="{i}" {"" if i==0 else "hidden"}><h3>{h}</h3><p>{content}</p></section>' for i,(h,content) in enumerate(notes))

page=r'''<!doctype html>
<html lang="zh-Hans"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Mini-SGLang Part II · 全链路流程图</title>
<style>
:root{color-scheme:light;--nav:#101c30;--paper:#f7faff;--ink:#14263b;--muted:#667890;--border:#dbe5ef}
*{box-sizing:border-box} body{margin:0;color:var(--ink);font-family:Inter,"Noto Sans CJK SC","Microsoft YaHei",Arial,sans-serif;background:#f5f7fb}
.layout{display:grid;grid-template-columns:292px minmax(0,1fr) 276px;height:100vh;min-height:600px;overflow:hidden}
.sidebar{background:var(--nav);color:#f5f9ff;padding:20px 14px;overflow:auto;min-height:0}
.brand{padding:7px 9px 18px;border-bottom:1px solid #2b3a53;margin-bottom:18px}
.brand .minor{font-weight:700;letter-spacing:1.8px;color:#68c9ef;font-size:11px}
h1{font-size:23px;line-height:1.3;margin:10px 0 8px;letter-spacing:-.4px}
.brand p{color:#a6b8cf;font-size:12px;line-height:1.8;margin:0}
.nav-item{display:block;text-align:left;width:100%;border:1px solid transparent;background:transparent;color:#b8cce0;border-radius:11px;margin:5px 0;padding:12px 13px;cursor:pointer;font:inherit;transition:.15s}
.nav-item:hover{background:#25354d;color:white}.nav-item.active{background:#203f64;border-color:#3877af;color:white}
.nav-item strong{font-size:15px;display:block;margin:4px 0}.nav-item small{font-size:11px;line-height:1.6;display:block;opacity:.72}
.nav-idx{font-size:10px;letter-spacing:1.1px;color:#83b5df}
.content{min-width:0;display:flex;flex-direction:column}
.toolbar{height:auto;min-height:72px;background:#fff;border-bottom:1px solid var(--border);padding:13px 23px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.toolbar h2{font-size:18px;margin:0 0 4px}.toolbar p{font-size:11px;color:#667890;margin:0}
.tools{display:flex;gap:7px;align-items:center;flex-wrap:wrap}
.tools button,.tools a{border:1px solid #cad8e6;border-radius:8px;background:white;color:#28476b;padding:9px 11px;cursor:pointer;font-size:12px;font-weight:650;text-decoration:none}
.tools button:hover,.tools a:hover{background:#eff7ff}
#stage{position:relative;flex:1;overflow:hidden;touch-action:none;cursor:grab;background-image:radial-gradient(#d6dfec .8px,transparent .8px);background-size:20px 20px;background-color:#f9fbff}
#stage.dragging{cursor:grabbing}
#stage svg{position:absolute;top:0;left:0;max-width:none;max-height:none;overflow:visible;transform-origin:0 0;user-select:none}
#stage svg a:hover polygon,#stage svg a:hover path,#stage svg a:hover rect{stroke:#ef8f19 !important;stroke-width:3px!important}
#stage svg a:hover text{font-weight:bold}
.hint{position:absolute;left:15px;bottom:13px;background:rgba(15,34,60,.78);color:white;font-size:11px;border-radius:7px;padding:9px 12px;pointer-events:none}
.infos{background:white;border-left:1px solid var(--border);overflow:auto;padding:21px 19px;font-size:13px}
.infos h2{font-size:16px;margin:0 0 13px}.infos h3{font-size:15px;margin:0 0 8px}.infos p{color:#4b5e70;line-height:1.9;margin:0 0 13px}
.info-slab{border-bottom:1px solid #e3eaf1;padding-bottom:18px;margin-bottom:17px}
.legend{display:grid;grid-template-columns:1fr 1fr;gap:11px 7px;margin-top:11px;font-size:12px}
.leg{display:flex;align-items:center;gap:7px;color:#35465b}.swatch{width:12px;height:12px;border-radius:3px;display:inline-block}
.source-list{display:grid;gap:9px;margin:12px 0}.source-list a{color:#246194;text-decoration:none;font-size:11px;line-height:1.4}.source-list a:hover{text-decoration:underline}
small.disclaimer{color:#738498;line-height:1.8;display:block;margin-top:15px}
@media(max-width:1100px){.layout{grid-template-columns:240px minmax(0,1fr)}.infos{display:none}}
@media(max-width:680px){.layout{grid-template-columns:1fr;grid-template-rows:auto 1fr;height:100dvh}.sidebar{display:flex;gap:8px;overflow:auto;padding:8px 10px}.brand{display:none}.nav-item{min-width:180px}.toolbar{padding:10px}.toolbar h2{font-size:15px}}
</style></head><body>
<div class="layout"><aside class="sidebar"><div class="brand"><div class="minor">DEEP SYSTEMS MAP · 2026</div><h1>Mini-SGLang<br/>Part II 全链路</h1><p>10 章知识融合 · 7 张可交互矢量流程图<br/>精确区分教学抽象与参考版本源码</p></div>
%%BUTTONS%%
</aside><main class="content"><div class="toolbar"><div><h2 id="head-title">总图 · End-to-End</h2><p id="head-subtitle">全部链路：HTTP → Scheduler → KV → Engine → Streaming / Abort</p></div><div class="tools">
<button id="zoom-in">＋ 放大</button><button id="zoom-out">－ 缩小</button><button id="fit">适应画布</button><button id="one-to-one">100%</button><a id="download-svg" href="" download="mini-sglang.svg">下载当前 SVG</a>
</div></div><div id="stage"><div class="hint">滚轮缩放 · 拖动画布 · 点击节点跳到对应 GitHub 章节</div></div></main>
<aside class="infos"><div class="info-slab"><h2>本页重点</h2>%%NOTES%%</div>
<div class="info-slab"><h2>分类图例</h2><div class="legend">
<div class="leg"><i class="swatch" style="background:#19669A"></i>HTTP / IO</div><div class="leg"><i class="swatch" style="background:#6C44B1"></i>Scheduler</div>
<div class="leg"><i class="swatch" style="background:#A65D13"></i>Memory / KV</div><div class="leg"><i class="swatch" style="background:#13855F"></i>GPU / Model</div>
<div class="leg"><i class="swatch" style="background:#AD4683"></i>Tensor Parallel</div><div class="leg"><i class="swatch" style="background:#00877B"></i>Speculative</div>
<div class="leg"><i class="swatch" style="background:#A87910"></i>Decision</div><div class="leg"><i class="swatch" style="background:#B83333"></i>Abort / Error</div>
</div></div>
<div class="info-slab"><h2>课程源码</h2><div class="source-list">%%SOURCES%%</div></div>
<small class="disclaimer">对应 datawhalechina/zero-to-sglang Part II。不是自动生成的 1:1 程序控制流图；合并相同职责、标明分支和概念。Ch10 为正式 SGLang 投机解码扩展参考。</small>
</aside></div>
%%TEMPLATES%%
<script>
const names=%%NAMES%%;
const files=%%FILES%%;
const stage=document.querySelector('#stage');let activeIndex=0,activeSvg=null;let tx=0,ty=0,zoom=1,baseW=100,baseH=100;let drag=null;
function apply(){if(activeSvg) activeSvg.style.transform=`translate(${tx}px,${ty}px) scale(${zoom})`}
function show(i){
 activeIndex=i;document.querySelectorAll('.nav-item').forEach(el=>el.classList.toggle('active',Number(el.dataset.index)===i));
 document.querySelectorAll('[data-info]').forEach(el=>el.hidden=Number(el.dataset.info)!==i);
 stage.querySelector('svg')?.remove();const root=document.querySelector('#diagram-'+i).content.querySelector('svg').cloneNode(true);
 activeSvg=root;stage.prepend(root);const vb=root.viewBox.baseVal;baseW=vb.width;baseH=vb.height;
 root.setAttribute('width',baseW);root.setAttribute('height',baseH);root.style.width=baseW+'px';root.style.height=baseH+'px';
 document.getElementById('head-title').textContent=names[i][0]+' · '+names[i][1];document.getElementById('head-subtitle').textContent=names[i][2];
 const data=new Blob([new XMLSerializer().serializeToString(root)],{type:'image/svg+xml'});const url=URL.createObjectURL(data);const a=document.getElementById('download-svg');if(a.dataset.url)URL.revokeObjectURL(a.dataset.url);a.dataset.url=url;a.href=url;a.download=files[i]+'.svg';
 requestAnimationFrame(fit);
}
function fit(){const w=stage.clientWidth,h=stage.clientHeight;zoom=Math.min((w-48)/baseW,(h-48)/baseH,1.8);tx=(w-baseW*zoom)/2;ty=(h-baseH*zoom)/2;apply()}
function zoomAt(factor,x,y){const z=Math.max(.06,Math.min(6,zoom*factor));tx=x-(x-tx)*z/zoom;ty=y-(y-ty)*z/zoom;zoom=z;apply()}
stage.addEventListener('wheel',e=>{e.preventDefault();const rect=stage.getBoundingClientRect();zoomAt(e.deltaY<0?1.14:1/1.14,e.clientX-rect.left,e.clientY-rect.top)},{passive:false});
stage.addEventListener('pointerdown',e=>{if(e.button!==0)return; if(e.target.closest?.('a'))return;drag={x:e.clientX,y:e.clientY};stage.setPointerCapture(e.pointerId);stage.classList.add('dragging')});
stage.addEventListener('pointermove',e=>{if(!drag)return;tx+=e.clientX-drag.x;ty+=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;apply()});
stage.addEventListener('pointerup',e=>{drag=null;stage.classList.remove('dragging')});stage.addEventListener('pointercancel',e=>{drag=null;stage.classList.remove('dragging')});
document.getElementById('zoom-in').onclick=()=>zoomAt(1.3,stage.clientWidth/2,stage.clientHeight/2);
document.getElementById('zoom-out').onclick=()=>zoomAt(1/1.3,stage.clientWidth/2,stage.clientHeight/2);
document.getElementById('fit').onclick=fit;
document.getElementById('one-to-one').onclick=()=>{zoom=1;tx=(stage.clientWidth-baseW)/2;ty=(stage.clientHeight-baseH)/2;apply()};
document.querySelectorAll('.nav-item').forEach(el=>el.onclick=()=>show(Number(el.dataset.index)));
window.addEventListener('resize',fit);show(0);
</script></body></html>'''
page=page.replace('%%BUTTONS%%',buttons).replace('%%NOTES%%',rightnotes).replace('%%SOURCES%%','\n'.join(chapterlinks)).replace('%%TEMPLATES%%',templates).replace('%%NAMES%%',json.dumps(labels,ensure_ascii=False)).replace('%%FILES%%',json.dumps(paths))
(OUT/'Mini-SGLang-PartII-Interactive.html').write_text(page,encoding='utf-8')
print('HTML',len(page),'bytes',flush=True)

# a chapter-by-chapter source index + README
md='''# Mini-SGLang Part II 全链路流程图｜使用说明

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

'''
for i,f in CH.items():
 md+=f'- [第 {i} 章：{f}]({BASE+quote(f)})\n'
md+='\n- [mini-sglang 源码](https://github.com/sgl-project/mini-sglang)\n- [正式 SGLang Speculative Decoding 参考](https://github.com/sgl-project/sglang/tree/1291f3175d3b7614821884e9653985fcd3c3dae8/python/sglang/srt/speculative)\n'
(OUT/'README-使用指南.md').write_text(md,encoding='utf-8')
print('Done')
