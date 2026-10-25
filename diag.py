import urllib.request, urllib.parse, json, time
from collections import Counter
UA={"User-Agent":"Mozilla/5.0"}
BASE="https://datasets-server.huggingface.co"
DS="lmarena-ai/leaderboard-dataset"
ds=urllib.parse.quote(DS)
def get(u, tries=5):
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60).read().decode())
        except Exception as e:
            if "429" in str(e) and i<tries-1:
                time.sleep(6*(i+1)); continue
            raise
CFGS=["agent","agent_bash_recovery_steps","agent_bash_reliability","agent_bash_reliability_no_prereq",
"agent_praise_complaint","agent_steerability","agent_task_outcome_explicit","agent_tool_hallucination",
"document","document_style_control","image_edit","image_to_video","search","search_factuality",
"search_style_control","text","text_factuality","text_style_control","text_to_image","text_to_video",
"video_edit","vision","vision_style_control","webdev"]
print("CFG|OVERALL|SUBS")
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
            time.sleep(1.2)
        overall=cats.get("overall",0)
        subs=sorted([k for k in cats if k!="overall"])
        print(f"{cfg}|{overall}|{subs}")
        time.sleep(1.5)
    except Exception as e:
        print(f"{cfg}|ERR|{str(e)[:60]}")
