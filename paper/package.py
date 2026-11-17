"""Package the bilingual manuscript, reproducible figures, and frozen evidence."""
from pathlib import Path
import hashlib
import json
import zipfile

P = Path(__file__).resolve().parent
ROOT = P.parent
files = [
    'docs/paper.md', 'docs/paper-中文.md', 'docs/paper-references.bib',
    'docs/树状种群与反思简化.md', 'docs/EvoTrace-v3设计.md', 'docs/EvoTrace-v3实验方案.md',
    'docs/记录/2026-11-16-EvoTrace对比任务筛选.md', 'docs/记录/README.md',
    'docs/figures/evotrace/README.md',
    'docs/figures/evotrace/12-v3-experience-cycle.png',
    'docs/figures/evotrace/13-v3-experience-accumulation.png',
    'docs/figures/evotrace/current-generation-prompts.json',
    'docs/figures/evotrace/current-generation-record.json',
    'paper/README.md', 'paper/build.py', 'paper/validate.py', 'paper/package.py',
    'paper/requirements.txt', 'paper/style.tex', 'paper/references.bib',
    'paper/main.tex', 'paper/main-zh.tex',
    'paper/trees.pdf', 'paper/trees-中文.pdf',
    'paper/collect_representative_evidence.py', 'paper/plot_representative_evidence.py',
    'paper/collect_evidence.py', 'paper/plot_evidence.py', 'paper/analyze_experience_reuse.py',
]
for directory in ['paper/evidence', 'paper/vendor/icml2027', 'paper/figures']:
    files.extend(str(path.relative_to(ROOT)) for path in (ROOT / directory).iterdir() if path.is_file())
files = sorted(set(files))
hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files}
destination = P / 'TREES-paper-source.zip'
with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(ROOT / path, 'TREES-paper/' + path)
    archive.writestr('TREES-paper/SHA256.json', json.dumps(hashes, indent=2, ensure_ascii=False) + '\n')
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    for path, digest in hashes.items():
        assert hashlib.sha256(archive.read('TREES-paper/' + path)).hexdigest() == digest
print(f'{destination}: {len(files)} files and SHA256.json; archive hashes verified.')
