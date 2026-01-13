import json
import csv
from pathlib import Path
import math

infile = Path('ragas_evaluation_results_20251026_011714.json')
outfile = Path('ragas_evaluation_results_20251026_011714_clean.csv')

if not infile.exists():
    print('Input JSON not found:', infile)
    raise SystemExit(1)

data = json.loads(infile.read_text(encoding='utf-8'))

fieldnames = [
    'idx', 'question', 'source_pdf', 'noinfo_answer',
    'faithfulness', 'faithfulness_status',
    'answer_correctness', 'answer_correctness_status',
    'nv_context_relevance', 'nv_context_relevance_status',
    'answer_relevancy', 'answer_relevancy_status'
]

with outfile.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i,item in enumerate(data):
        q = item.get('question') or item.get('user_input') or ''
        src = item.get('source_pdf','')
        ans = (item.get('answer') or item.get('response') or '').strip()
        noinfo = any(k in ans.lower() for k in ['unfortunately','does not mention','no mention','not explicitly'])

        def fmt_metric(key):
            val = item.get(key)
            if val is None:
                return '', 'missing'
            try:
                v = float(val)
                if math.isnan(v):
                    return '', 'nan'
                return f"{v:.6f}", 'present'
            except Exception:
                return str(val), 'present'

        f_f, s_f = fmt_metric('faithfulness')
        f_ac, s_ac = fmt_metric('answer_correctness')
        f_nv, s_nv = fmt_metric('nv_context_relevance')
        f_ar, s_ar = fmt_metric('answer_relevancy')

        writer.writerow({
            'idx': i+1,
            'question': q,
            'source_pdf': src,
            'noinfo_answer': 'yes' if noinfo else 'no',
            'faithfulness': f_f,
            'faithfulness_status': s_f,
            'answer_correctness': f_ac,
            'answer_correctness_status': s_ac,
            'nv_context_relevance': f_nv,
            'nv_context_relevance_status': s_nv,
            'answer_relevancy': f_ar,
            'answer_relevancy_status': s_ar
        })

print('Wrote cleaned CSV to:', outfile)
