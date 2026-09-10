"""Load original official tables and execute the saved report evidence queries.

The separate analyze.py verifies these official values against released predictions
and truth. Here SQLite provides directly auditable official-table queries for the
report reader, which requires SQL provenance for native charts/tables.
"""
import json
from pathlib import Path
import sqlite3
import openpyxl
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCES = HERE.parents[3] / 'pwscup2026-results-publication/prelim/results/20260909/sources'
wb = openpyxl.load_workbook(SOURCES / '公開用結果表_20260909.xlsx', data_only=True, read_only=True)
db = sqlite3.connect(':memory:')
for sheet, table, names in [
    ('攻撃_標的別_MIA', 'official_mia', ['attacker','target','members','rare_members','tpr_all','tpr_rare','t_j','auc']),
    ('攻撃_標的別_AIA', 'official_aia', ['attacker','target','r_all','r_rare','r_rare_raw','p_m','p_c','p_m_rare','p_c_rare','loss']),
]:
    rows = list(wb[sheet].values)
    pd.DataFrame(rows[1:], columns=names).to_sql(table, db, index=False)
pd.read_csv(SOURCES / 'ground_truth/aia_truth_24.csv').to_sql('aia_truth_24', db, index=False)
sql = (HERE / 'official_tables.sql').read_text(encoding='utf-8')
queries = [s.strip() + ';' for s in sql.split(';') if s.strip()]
assert len(queries) == 3
datasets = {name: json.loads(pd.read_sql_query(q, db).to_json(orient='records', double_precision=15))
            for name, q in zip(['mia_positive','rare','aia'], queries)}
analysis = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
for row in datasets['mia_positive']:
    a = next(x for x in analysis['MIA']['attackers'] if x['attacker']==row['attacker_id'])
    assert abs(a['score'] - row['tpr']) < 1e-12
for row in datasets['aia']:
    a = (analysis['AIA']['constant_10_year_baseline'] if row['method']=='全員に10年'
         else next(x for x in analysis['AIA']['attackers'] if x['attacker']==int(row['method'][3:])))
    assert row['m_correct']==a['member_correct'] and row['c_correct']==a['control_correct']
    assert abs(row['loss']-a['score']) < 1e-12
(HERE / 'report_data.json').write_text(json.dumps({'datasets': datasets, 'queries':queries}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Official SQL tables reconcile with the independently rescored results.')
