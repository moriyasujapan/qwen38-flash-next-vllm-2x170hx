import json,time,urllib.request,sys
import os
B=os.environ.get("BASE","http://127.0.0.1:18024")+"/v1/chat/completions"
p=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"report-prompt-ja.txt")).read()
body={"model":"flash-next-w4a16","messages":[{"role":"user","content":p}],
      "max_tokens":16000,"temperature":0.7,"top_p":0.95,"stream":True,
      "stream_options":{"include_usage":True},
      "chat_template_kwargs":{"reasoning_effort":"medium"}}
req=urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
t0=time.time();tf=None;out=[];reas=0;toks=0;fin=None
for raw in urllib.request.urlopen(req,timeout=3600):
    line=raw.decode().strip()
    if not line.startswith("data: ") or line=="data: [DONE]": continue
    ev=json.loads(line[6:])
    if ev.get("usage"): toks=ev["usage"]["completion_tokens"]; rt=ev["usage"].get("completion_tokens_details",{}) or {}
    ch=ev.get("choices") or []
    if not ch: continue
    d=ch[0]["delta"]
    if ch[0].get("finish_reason"): fin=ch[0]["finish_reason"]
    if d.get("reasoning_content"): reas+=len(d["reasoning_content"]); tf=tf or time.time(); tl=time.time()
    if d.get("content"): out.append(d["content"]); tf=tf or time.time(); tl=time.time()
txt="".join(out)
open("report.md","w").write(txt)
print(f"finish={fin} TTFT={tf-t0:.1f}s wall={tl-t0:.1f}s completion_tokens={toks} decode={(toks-1)/(tl-tf):.1f} tok/s")
print(f"reasoning chars={reas}  answer chars={len(txt)}")
