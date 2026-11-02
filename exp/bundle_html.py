"""세 HTML을 하나로 묶는다. 원본은 그대로 둔다."""
import re
from pathlib import Path

BASE = Path("ppt/0. paper")
FILES = [("how", "howtoreadpaper.html", "논문 읽는 법"),
         ("papers", "paper_summary.html", "관련연구 분석"),
         ("concepts", "concepts.html", "개념 설명"),
         ("log", "experiment_log.html", "실험 진행 기록")]

css, bodies = [], []
for anchor, name, label in FILES:
    t = (BASE / name).read_text(encoding="utf-8")
    css.append(re.search(r"<style>(.*?)</style>", t, re.S).group(1))
    main = re.search(r"<main[^>]*>(.*?)</main>", t, re.S).group(1)
    main = re.sub(r"<footer>.*?</footer>", "", main, flags=re.S)
    bodies.append((anchor, label, main))

extra = """
body{background:var(--paper)}
main{max-width:980px;margin:0 auto;padding:28px 24px 96px}
section{scroll-margin-top:76px}
nav.toc{position:sticky;top:0;z-index:20;background:#fff;
  border-bottom:1px solid var(--rule);padding:12px 16px;
  display:flex;gap:10px;justify-content:center;flex-wrap:wrap;
  font-family:'Malgun Gothic',system-ui,sans-serif;font-size:15px}
nav.toc a{color:var(--navy);text-decoration:none;font-weight:700;
  border:1px solid var(--rule);border-radius:999px;padding:6px 14px}
nav.toc a:hover{border-color:var(--gold);color:var(--gold)}
nav.toc a.top{background:var(--navy);color:#fff;border-color:var(--navy)}
hr.sep{border:0;border-top:2px solid var(--rule);margin:72px 0 8px}
.meta{color:var(--muted);font-size:13px;text-align:center;padding:10px 16px}
"""

nav = "".join(f'<a href="#{a}">{l}</a>' for a, l, _ in bodies)
secs = []
for i, (anchor, label, body) in enumerate(bodies):
    sep = '<hr class="sep">' if i else ""
    secs.append(f'{sep}<section id="{anchor}">{body}</section>')

html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SESSIONCONTROL · 문서 모음 (단일 파일)</title>
<style>
{chr(10).join(css)}
{extra}
</style>
</head>
<body>
<nav class="toc">{nav}<a class="top" href="#top">맨 위</a></nav>
<main id="top">
{''.join(secs)}
</main>
<div class="meta">단일 파일 번들 · 외부 리소스 없음 · 원본: howtoreadpaper.html, paper_summary.html, concepts.html, experiment_log.html</div>
</body>
</html>
"""

out = BASE / "all-in-one.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, f"{out.stat().st_size/1024:.1f}KB")
