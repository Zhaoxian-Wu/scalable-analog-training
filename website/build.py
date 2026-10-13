"""Copy the static homepage to dist/ without third-party dependencies."""
from pathlib import Path
import hashlib
import re
import shutil

root = Path(__file__).resolve().parent
output = root / "dist"
output.mkdir(exist_ok=True)

# Refresh cached diagrams when their contents change, in both local entry points.
index = root / "index.html"
markup = index.read_text()
asset_reference = re.compile(
    r'((?:src|href)=")(assets/(?:interactive/diagrams\.(?:js|css)|'
    r'gradient-[^"?]+\.(?:svg|json)))(?:\?[^"\s]*)?(")'
)


def version_asset(match):
    asset = root / match[2]
    version = hashlib.sha256(asset.read_bytes()).hexdigest()[:12]
    return f'{match[1]}{match[2]}?v={version}{match[3]}'


versioned_markup = asset_reference.sub(version_asset, markup)
if versioned_markup != markup:
    index.write_text(versioned_markup)

for filename in ("index.html", "styles.css", "interactives.css", "app.js", "training-scale.js", "THIRD_PARTY_NOTICES.txt"):
    shutil.copy2(root / filename, output / filename)
shutil.copytree(root / "assets", output / "assets", dirs_exist_ok=True)
print(f"Static site ready: {output}")
