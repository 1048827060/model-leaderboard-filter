import urllib.request, urllib.parse, json, ssl, os
UA={"User-Agent":"Mozilla/5.0"}
BASE="https://datasets-server.huggingface.co"
DS="lmarena-ai/leaderboard-dataset"
def get(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60).read().decode())
ds=urllib.parse.quote(DS)
print("### SPLITS/CONFIGS ###")
try:
    s=get(f"{BASE}/splits?dataset={ds}")
    cfgs={}
    for x in s.get("splits",[]):
        if x.get("config"): cfgs.setdefault(x["config"],[]).append(x.get("split"))
    for c,sp in cfgs.items(): print("CONFIG:",c,"| splits:",sp)
except Exception as e:
    print("splits err:",e)
print()
print("### CATEGORIES PER CONFIG (前 800 行采样) ###")
for cfg in ["text","webdev","vision"]:
    try:
        d=get(f"{BASE}/rows?dataset={ds}&config={cfg}&split=latest&offset=0&length=100")
        feats=[f["name"] for f in d.get("features",[])]
        catcol=[f for f in feats if "categor" in f.lower()]
        rows=[r["row"] for r in d.get("rows",[])]
        for p in range(1,8):
            if len(rows)>=800: break
            d2=get(f"{BASE}/rows?dataset={ds}&config={cfg}&split=latest&offset={p*100}&length=100")
            rows+= [r["row"] for r in d2.get("rows",[])]
        from collections import Counter
        c=Counter()
        for r in rows:
            if catcol: c[str(r.get(catcol[0]))]+=1
        print(f"--- {cfg} | cols={feats}")
        print(f"    catcol={catcol} | distinct={len(c)}")
        for k,v in c.most_common(30): print(f"      {k}: {v}")
    except Exception as e:
        print(f"--- {cfg} err:",str(e)[:200])
