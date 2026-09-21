import json,time,urllib.request,sys
import os
B=os.environ.get("BASE","http://127.0.0.1:18024")+"/v1/chat/completions"
def run(prompt,maxtok=800):
    body={"model":"flash-next-w4a16","messages":[{"role":"user","content":prompt}],
          "max_tokens":maxtok,"temperature":0.7,"top_p":0.95,"stream":True,
          "stream_options":{"include_usage":True},
          "chat_template_kwargs":{"reasoning_effort":"medium"}}
    req=urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.time();tfirst=None;toks=0
    for raw in urllib.request.urlopen(req,timeout=1800):
        line=raw.decode().strip()
        if not line.startswith("data: ") or line=="data: [DONE]": continue
        ev=json.loads(line[6:])
        if ev.get("usage"): toks=ev["usage"]["completion_tokens"]
        ch=ev.get("choices") or []
        if ch and (ch[0]["delta"].get("content") or ch[0]["delta"].get("reasoning_content")):
            if tfirst is None: tfirst=time.time()
            tlast=time.time()
    dec=(toks-1)/(tlast-tfirst) if tfirst and toks>1 else 0
    print(f"  TTFT {tfirst-t0:.2f}s, {toks} tok, decode {dec:.1f} tok/s")
    return dec
print("run 1 (short prose):"); a=run("量子もつれを3文で説明して")
print("run 2 (code):");       b=run("Pythonで二分探索を書いて。コードだけ、説明不要。")
print("run 3 (longer):");     c=run("ローカルLLMを自宅で動かす利点と欠点を、それぞれ5つずつ挙げて")
print(f"median-ish decode: {sorted([a,b,c])[1]:.1f} tok/s")
