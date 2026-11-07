import svgwrite,subprocess
from pathlib import Path
P=Path('/mnt/data/minisglang_flowcharts');W,H=1960,1160
s=svgwrite.Drawing(str(P/'00_roadmap_overview.svg'),size=(W,H),viewBox=f'0 0 {W} {H}')
s.add(s.rect(insert=(0,0),size=(W,H),fill='#0B1423'))
# subtle ruled background
for y in range(0,H,32):s.add(s.line((0,y),(W,y),stroke='#13253B',stroke_width=.5))
style={'font_family':'Noto Sans CJK SC, sans-serif'}
def txt(x,y,z,size=23,color='#E8F3FF',weight='normal'):
 s.add(s.text(z,insert=(x,y),font_family='Noto Sans CJK SC',font_size=size,fill=color,font_weight=weight))
txt(80,88,'MINI-SGLANG  |  PART II · END-TO-END',42,'#F3F8FF','bold')
txt(80,133,'一条请求，串通 10 章 · 精简阅读路线（完整版见互动流程图）',23,'#9CB8D6')
items=[
 ('01','HTTP API / uid','FastAPI → 解析请求','Ch 2 / 5','#1D79BE'),
 ('02','Tokenizer','Chat Template → input_ids','Ch 2 / 5','#1D79BE'),
 ('03','Scheduler 循环','waiting / running / 组批','Ch 1 / 6','#815CD8'),
 ('04','Prefix Admission','Radix 匹配、lock、资源检查','Ch 6 / 8','#C8892D'),
 ('05','Paged KV Allocator','free pages → page_table / out_loc','Ch 4 / 7','#C8892D'),
 ('06','GPU Forward + TP','QKV → Attention → MLP → LM Head','Ch 3 / 4 / 9','#27A884'),
 ('07','Sampler / Spec','普通采样；或 Draft + Verify','Ch 3 / 10','#31B6C3'),
 ('08','Req State & Stop','追加 token、EOS / max_tokens','Ch 2 / 6','#815CD8'),
 ('09','Detokenizer','token ID → 增量文字','Ch 2 / 5','#1D79BE'),
 ('10','API Frontend','UserReply / ack_map / event_map','Ch 5','#1D79BE'),
 ('11','SSE / JSON','流式 deltas 或完整文本','Ch 5','#1D79BE'),
 ('12','Client + Cleanup','[DONE] / cache_req / free','Ch 6 / 8','#C8892D'),
]
# snake placement
locations=[(0,0),(1,0),(2,0),(3,0),(3,1),(2,1),(1,1),(0,1),(0,2),(1,2),(2,2),(3,2)]
cardw,cardh=420,213; gapx=36; gapy=75;x0=70;y0=199
for i,(num,title,descr,ch,col) in enumerate(items):
 c,r=locations[i];x=x0+c*(cardw+gapx);y=y0+r*(cardh+gapy)
 s.add(s.rect(insert=(x,y),size=(cardw,cardh),rx=18,ry=18,fill='#111F32',stroke=col,stroke_width=2.4))
 s.add(s.rect(insert=(x,y),size=(cardw,9),rx=5,ry=5,fill=col))
 txt(x+25,y+47,num,20,col,'bold');txt(x+25,y+94,title,29,'#F4F8FE','bold');txt(x+25,y+138,descr,18,'#B6C7D9');txt(x+25,y+183,ch,17,col,'bold')
# connections: arrow ornaments placed between boxes
marker=s.marker(id='arr',insert=(8,4),size=(10,8),orient='auto')
marker.add(s.path(d='M0,0 L8,4 L0,8 Z',fill='#8EACC6'));s.defs.add(marker)
for i in range(11):
 c1,r1=locations[i];c2,r2=locations[i+1]
 x1=x0+c1*(cardw+gapx);y1=y0+r1*(cardh+gapy)
 x2=x0+c2*(cardw+gapx);y2=y0+r2*(cardh+gapy)
 if r1==r2:
  if c2>c1: p=((x1+cardw+5,y1+cardh/2),(x2-8,y2+cardh/2))
  else: p=((x1-5,y1+cardh/2),(x2+cardw+8,y2+cardh/2))
 else:p=((x1+cardw/2,y1+cardh+6),(x2+cardw/2,y2-8))
 s.add(s.line(p[0],p[1],stroke='#8EACC6',stroke_width=3,marker_end=marker.get_funciri()))
# dashed feedback arrow no intersections with card box
s.add(s.path(d='M 70 515 C 23 515 19 505 19 458 L 19 175 L 1034 175 L 1034 195',fill='none',stroke='#B88DF5',stroke_width=2.5,stroke_dasharray='8,8',marker_end=marker.get_funciri()))
txt(80,1067,'关键回路：未完成的请求返回 Scheduler；Chunked Prefill 先补齐 Prompt；结束或取消后归档可复用 KV 并释放其余页。',22,'#BFD1E7')
txt(80,1110,'* 投机解码是正式 SGLang 的扩展路径，并非 Mini-SGLang 教程基线默认实现。',18,'#90A9C6')
s.save()
subprocess.run(['inkscape',str(P/'00_roadmap_overview.svg'),'--export-type=png','--export-filename='+str(P/'00_roadmap_overview.png'),'--export-width=1960'],check=True,capture_output=True)
print('done', (P/'00_roadmap_overview.png').stat().st_size)
