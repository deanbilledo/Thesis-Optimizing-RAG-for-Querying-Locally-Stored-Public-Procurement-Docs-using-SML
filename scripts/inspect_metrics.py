import json
from pathlib import Path

data = json.loads(Path('ragas_evaluation_results_20251026_011714.json').read_text(encoding='utf-8'))

for i,item in enumerate(data):
    q = item.get('question') or item.get('user_input')
    present = {k: (item.get(k) is not None) for k in ['faithfulness','answer_correctness','nv_context_relevance','answer_relevancy']}
    noinfo = False
    resp = (item.get('answer') or item.get('response') or '').lower()
    if 'unfortunately' in resp or 'does not mention' in resp or 'no mention' in resp or 'not explicitly' in resp:
        noinfo = True
    print(f"{i+1:02d}: present={present} noinfo={noinfo} source_pdf={item.get('source_pdf')}")
