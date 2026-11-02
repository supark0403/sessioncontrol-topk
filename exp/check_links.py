import re

for p in ("ppt/0. paper/concepts.html", "ppt/0. paper/paper_summary.html"):
    t = open(p, encoding="utf-8").read()
    print("===", p)
    for m in re.finditer(r'href="([^"]+)"', t):
        print("  href:", m.group(1))
    for m in re.finditer(r"<code>([^<]*)</code>", t):
        s = m.group(1)
        if "paper" in s or "concept" in s or "howto" in s:
            print("  code:", s)
