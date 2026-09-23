"""Freeze reviewed code hashes before the first private reservation."""
import json
from release20 import HERE,CORE,SOURCES,sha,read
if (HERE/'release_ledger/reservation.json').exists():
    raise RuntimeError('Cannot change configuration after reservation')
config=read(HERE/'config20.json')
config['source_sha256']={n:sha(CORE/n) for n in SOURCES}
(HERE/'config20.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
print('Reviewed source hashes frozen; no protected data read.')
