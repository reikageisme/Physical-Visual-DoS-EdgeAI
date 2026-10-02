"""Prepare a deterministic COCO val2017 subset without changing existing data.

Only public COCO annotations/images are downloaded. No camera or private files.
The subset is held out from our optimization; upstream weight-selection use is
unknown, so it must not be called a completely unseen distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, target):
    target = Path(target)
    if target.exists():
        return
    temporary = target.with_suffix(target.suffix + '.part')
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Research-validation/1.0'})
            with urllib.request.urlopen(req, timeout=90) as response, temporary.open('wb') as f:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    f.write(block)
            temporary.replace(target)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def download_member(url, member, target):
    """Read one public ZIP member with verified HTTP ranges, not the 241 MB ZIP."""
    def read_range(start, end):
        request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 206:
                raise RuntimeError('Origin did not honor the requested byte range')
            data = response.read()
            if len(data) != end-start+1:
                raise RuntimeError('Incomplete ZIP byte range')
            return data
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
        length = int(response.headers['Content-Length'])
    tail = read_range(max(0, length-65557), length-1)
    position = tail.rfind(b'PK\x05\x06')
    if position < 0:
        raise RuntimeError('Missing ZIP directory')
    end = struct.unpack_from('<4s4H2LH', tail, position)
    cd = read_range(end[6], end[6]+end[5]-1)
    offset = 0
    while offset < len(cd):
        entry = struct.unpack_from('<4s6H3L5H2L', cd, offset)
        assert entry[0] == b'PK\x01\x02'
        name_length, extra_length, comment_length = entry[10:13]
        name = cd[offset+46:offset+46+name_length].decode('utf-8')
        if name == member:
            local_offset = entry[-1]
            local = struct.unpack('<4s5H3L2H', read_range(local_offset, local_offset+29))
            begin = local_offset+30+local[-2]+local[-1]
            compressed = read_range(begin, begin+entry[8]-1)
            data = zlib.decompress(compressed, -15) if entry[4] == 8 else compressed
            if len(data) != entry[9] or zlib.crc32(data) != entry[7]:
                raise RuntimeError('ZIP member CRC/length mismatch')
            Path(target).write_bytes(data)
            return
        offset += 46+name_length+extra_length+comment_length
    raise KeyError(member)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=int, default=128)
    args = parser.parse_args()
    out = ROOT / 'outputs/datasets/coco_val2017_extended'
    out.mkdir(parents=True, exist_ok=True)
    archive = out / 'annotations_trainval2017.zip'
    annotation = out / 'instances_val2017.json'
    if not annotation.exists():
        print('Downloading official val annotation member using ZIP byte ranges', flush=True)
        download_member('https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip',
                        'annotations/instances_val2017.json', annotation)
    source = json.loads(annotation.read_text(encoding='utf-8'))
    selected = sorted(source['images'], key=lambda row: row['id'])[:args.images]
    ids = {row['id'] for row in selected}
    training = ROOT / 'outputs/datasets/coco128/images/train2017'
    training_stems = {p.stem for p in training.glob('*.jpg')}
    assert not training_stems.intersection(Path(row['file_name']).stem for row in selected)
    folder = out / 'images'
    folder.mkdir(exist_ok=True)

    def fetch(row):
        filename = row['file_name']
        url = row['coco_url'].replace('http://images.cocodataset.org/', 'https://s3.amazonaws.com/images.cocodataset.org/')
        download(url, folder / filename)
        return {'image_id': row['id'], 'file_name': filename, 'url': url,
                'sha256': digest(folder / filename), 'width': row['width'], 'height': row['height']}

    with ThreadPoolExecutor(max_workers=4) as pool:
        images = list(pool.map(fetch, selected))
    subset = {**{k: v for k, v in source.items() if k not in ('images', 'annotations')},
              'images': selected, 'annotations': [a for a in source['annotations'] if a['image_id'] in ids]}
    (out / 'instances_subset.json').write_text(json.dumps(subset, ensure_ascii=False), encoding='utf-8')
    manifest = {'source': 'COCO val2017', 'selection': 'first N image IDs in ascending order, fixed before new optimization',
                'images_count': len(images), 'images': images,
                'annotation_sha256': digest(annotation), 'subset_sha256': digest(out / 'instances_subset.json'),
                'optimization_overlap_image_ids': [], 'upstream_model_selection_overlap': 'unknown',
                'script_sha256': digest(__file__)}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'PREPARED {len(images)} images, {len(subset["annotations"])} official annotations', flush=True)


if __name__ == '__main__':
    main()
