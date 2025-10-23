from pathlib import Path
import importlib.util
import pandas as pd

p = Path(__file__).resolve().parents[1] / 'app.py'
spec = importlib.util.spec_from_file_location('app_mod', str(p))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('IMPORT_OK')
df = mod.load_data(mod.DATA_DIR)
jobs = df[df['type'] == 'job']
print('jobs rows:', len(jobs))
for i, r in jobs.head(10).iterrows():
    vals = {k: r.get(k) for k in ('job_url_direct', 'job_url', 'link', 'url')}
    link = None
    for key in ('job_url_direct', 'job_url', 'link', 'url'):
        if key in r.index:
            val = r.get(key)
            if pd.notna(val):
                s = str(val).strip()
                if s:
                    if s.lower().startswith('http'):
                        link = s
                        break
                    if link is None:
                        link = s
    print(i, 'raw:', vals)
    print('resolved link ->', link)
