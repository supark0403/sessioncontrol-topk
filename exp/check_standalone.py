import re
from pathlib import Path

for p in sorted(Path("ppt/0. paper").glob("*.html")):
    t = p.read_text(encoding="utf-8")
    hrefs = re.findall(r'href="([^"]+)"', t)
    srcs = re.findall(r'src="([^"]+)"', t)
    urls = re.findall(r'url\(\s*["\']?([^)"\']+)', t)
    fonts = re.findall(r'fonts\.(?:googleapis|gstatic)\.com', t)
    print(f"{p.name:24} href={hrefs} src={srcs} url={urls[:3]} googlefont={bool(fonts)}")
