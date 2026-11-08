"""Temporary affinity separation of own ChatGPT tree; restore in finally.

No process termination or priority/system changes. Worker itself pins CPU2.
Mutation journal is flushed BEFORE each affinity operation for crash recovery.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import psutil


def restricted(mask, excluded=(2,3)):
    result=sorted(set(mask)-set(excluded))
    if not result:
        raise RuntimeError('cannot exclude benchmark core without empty affinity')
    return result


def app_root():
    for p in psutil.Process().parents():
        if p.name().lower()=='chatgpt.exe':
            if p.username()!=psutil.Process().username():
                raise RuntimeError('ChatGPT ancestor owned by another user')
            return p
    raise RuntimeError('no own ChatGPT ancestor; no affinity changes')


class Separation:
    def __init__(self,path):
        self.root=app_root()
        self.records={}
        self.file=path.open('x',encoding='utf-8')
        self.write(dict(kind='root',pid=self.root.pid,created=self.root.create_time(),name=self.root.name()))

    def write(self,row):
        self.file.write(json.dumps(row)+'\n');self.file.flush()

    def apply(self):
        for attempt in range(3):
            for p in [self.root,*self.root.children(recursive=True)]:
                try:
                    key=(p.pid,p.create_time())
                    current=sorted(p.cpu_affinity())
                    if key in self.records:
                        if current!=self.records[key]['new']:
                            raise RuntimeError('affinity changed during preparation')
                        continue
                    new=restricted(current)
                    row=dict(kind='intent',pid=p.pid,created=key[1],name=p.name(),old=current,new=new)
                    self.write(row);self.records[key]=row
                    p.cpu_affinity(new)
                    if sorted(p.cpu_affinity())!=new:
                        raise RuntimeError('helper affinity readback failed')
                    self.write(dict(kind='applied',pid=p.pid,created=key[1]))
                except psutil.NoSuchProcess:
                    continue
            live=[self.root,*self.root.children(recursive=True)]
            if all(not set(p.cpu_affinity()) & {2,3} for p in live if p.is_running()):
                self.write(dict(kind='ready',processes=len(self.records),attempt=attempt))
                return
        raise RuntimeError('process tree changed too quickly; no benchmark')

    def restore(self):
        errors=0
        # Restore newly born helpers only if their mask equals the inherited
        # restricted mask. Never overwrite an unrelated/manual mask change.
        for p in [self.root,*self.root.children(recursive=True)]:
            try:
                key=(p.pid,p.create_time())
                if key not in self.records:
                    ancestors=[a for a in p.parents() if (a.pid,a.create_time()) in self.records]
                    if not ancestors:
                        continue
                    parent=self.records[ancestors[0].pid,ancestors[0].create_time()]
                    if sorted(p.cpu_affinity())==parent['new']:
                        self.records[key]=dict(pid=p.pid,created=key[1],name=p.name(),old=parent['old'],new=parent['new'])
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                continue
        for (pid,created),row in reversed(list(self.records.items())):
            try:
                p=psutil.Process(pid)
                if p.create_time()!=created:
                    self.write(dict(kind='restore_skipped_pid_reuse',pid=pid));continue
                current=sorted(p.cpu_affinity())
                if current not in [row['new'],row['old']]:
                    self.write(dict(kind='restore_skipped_changed_mask',pid=pid,current=current));continue
                if current!=row['old']:
                    p.cpu_affinity(row['old'])
                back=sorted(p.cpu_affinity())
                self.write(dict(kind='restored',pid=pid,readback=back,expected=row['old']))
                if back!=row['old']:
                    errors+=1
            except psutil.NoSuchProcess:
                self.write(dict(kind='exited',pid=pid))
            except Exception as e:
                self.write(dict(kind='restore_error',pid=pid,error=str(e)))
                errors+=1
        self.file.close()
        if errors:
            raise RuntimeError('app affinity restore has unresolved errors; inspect journal')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--tag',required=True);ap.add_argument('--commit',required=True)
    ap.add_argument('--user-idle',required=True)
    a=ap.parse_args()
    spec=importlib.util.spec_from_file_location('laptop_runner',Path(__file__).with_name('run.py'))
    run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
    run.load_lock(a.lock)
    journal=a.run_dir.with_name(a.run_dir.name+'_affinity.jsonl')
    sep=Separation(journal)
    try:
        sep.apply()
        run.campaign(a)  # blocks; fresh worker restores its own CPU2 explicitly
    finally:
        sep.restore()
    print('APP_AFFINITY_RESTORE_CHECK_FINISHED',flush=True)


if __name__=='__main__':main()
