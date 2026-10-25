import urllib.request, urllib.parse, json
from collections import Counter
UA={"User-Agent":"Mozilla/5.0"}
BASE="https://datasets-server.huggingface.co"
DS="lmarena-ai/leaderboard-dataset"
ds=urllib.parse.quote(DS)
def get(u): return json.loads(urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60).read().decode())
CFGS=["agent","agent_bash_recovery_steps","agent_bash_reliability","agent_bash_reliability_no_prereq",
"agent_praise_complaint","agent_steerability","agent_task_outcome_explicit","agent_tool_hallucination",
"document","document_style_control","image_edit","image_to_video","search","search_factuality",
"search_style_control","text","text_factuality","text_style_control","text_to_image","text_to_video",
"video_edit","vision","vision_style_control","webdev"]
print(f"{'config':<36} {'总榜模型数':>8}  {'分项category'}")
for cfg in CFGS:
    try:
        rows=[]; cats=Counter()
        for p in range(12):
            d=get(f"{BASE}/rows?dataset={ds}&config={cfg}&split=latest&offset={p*100}&length=100")
            rr=d.get("rows",[])
            if not rr: break
            for r in rr:
                row=r["row"]; rows.append(row); cats[str(row.get("category"))]+=1
            if len(rr)<100: break
        overall=cats.get("overall",0)
        subs=[k for k in cats if k!="overall"]
        print(f"{cfg:<36} {overall:>8}  {sorted(subs)}")
    except Exception as e:
        print(f"{cfg:<36} ERROR {str(e)[:80]}")
