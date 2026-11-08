"""Kiểm cấu trúc nhanh cho main.tex (không cần TeX): trích dẫn có trong .bib, môi trường cân bằng, nhãn nội bộ bị cấm, chỗ trống dữ liệu.
    python paper_en/check_manuscript.py
"""
import re
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent
def expand_inputs(path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError(f'cyclic LaTeX input: {path}')
    source = path.read_text(encoding='utf-8')
    def replace(match):
        child = Path(match.group(1))
        if not child.suffix:
            child = child.with_suffix('.tex')
        # TeX input paths resolve from the build directory, including nested inputs.
        child = D / child if (D / child).exists() else path.parent / child
        return expand_inputs(child, (*stack, path))
    return re.sub(r'\\input\{([^}]+)\}', replace, source)

t = expand_inputs(D / 'main.tex')
b = (D / 'references.bib').read_text(encoding='utf-8')

cites = set()
for m in re.findall(r'\\cite\{([^}]*)\}', t):
    cites.update(x.strip() for x in m.split(','))
keys = set(re.findall(r'@\w+\{([^,]+),', b))
print('citations used:', len(cites), '| bib entries:', len(keys))
print('  cited but missing in bib:', sorted(cites - keys))
print('  in bib but never cited  :', sorted(keys - cites))

begins = Counter(re.findall(r'\\begin\{(\w+\*?)\}', t))
ends = Counter(re.findall(r'\\end\{(\w+\*?)\}', t))
bad = {k: (begins[k], ends[k]) for k in set(begins) | set(ends) if begins[k] != ends[k]}
print('environments balanced:', not bad, bad or '')
print('braces balance:', t.count('{') - t.count('}'), '| odd dollar count:', t.count('$') % 2)
labels = re.findall(r'\\label\{([^}]*)\}', t)
refs = set(re.findall(r'\\ref\{([^}]*)\}', t))
print('labels duplicated:', [k for k, v in Counter(labels).items() if v > 1], '| refs without label:', sorted(refs - set(labels)))
print('NO DATA placeholders:', len(re.findall(r'\\nodata\{', t)))
banned = ['OLD128', 'X1-500', 'outputs/', 'R3', 'X2', 'exploit', 'vulnerab', 'exhaust', 'denial', 'Edge Server', 'edge server', 'physical']
for w in banned:
    n = len(re.findall(re.escape(w), t))
    if n:
        print(f'  banned/internal term "{w}": {n}')
print('words (approx, without LaTeX commands):', len(re.sub(r'\\[a-zA-Z]+|[{}$%&]', ' ', t).split()))
