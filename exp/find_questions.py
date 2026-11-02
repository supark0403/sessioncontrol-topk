import os
import re

pat = re.compile(r'((?:What|Where|How|When|Who|Which)[^"<>{}\n]{10,90}\?)')
seen, out = set(), []
for dp, _, fs in os.walk("code/docs"):
    for f in fs:
        p = os.path.join(dp, f)
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for m in pat.finditer(t):
            q = m.group(1).strip()
            if q in seen:
                continue
            seen.add(q)
            out.append((p, q))
print("unique questions:", len(out))
for p, q in out[:20]:
    print(p, "|", q)
