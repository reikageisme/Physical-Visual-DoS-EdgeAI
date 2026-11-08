"""Profile CPU thứ hai (Fedora, CHUẨN BỊ, chưa chạy): CHỈ ĐỌC cấu trúc CPU và trạng thái tần số, in profile đề xuất.

    python3 studies/linux_profile/topology_fedora.py > fedora_topology.json

Không ghi vào /sys, không đổi governor. Đề xuất profile 'P1_cpu2' = CPU logic đầu tiên của lõi vật lý thứ hai
(giống laptop Windows: tránh lõi 0, chỉ một luồng của lõi đó). Người dùng tự đặt governor/tần số theo README.
"""
import glob
import json
import os
import re


def read(p):
    try:
        return open(p).read().strip()
    except OSError as e:
        return f'khong_doc_duoc:{type(e).__name__}'


def main():
    cpus = sorted(int(re.findall(r'cpu(\d+)$', d)[0]) for d in glob.glob('/sys/devices/system/cpu/cpu[0-9]*'))
    info = {}
    for c in cpus:
        b = f'/sys/devices/system/cpu/cpu{c}'
        info[c] = dict(core_id=read(f'{b}/topology/core_id'), package=read(f'{b}/topology/physical_package_id'),
                       siblings=read(f'{b}/topology/thread_siblings_list'), governor=read(f'{b}/cpufreq/scaling_governor'),
                       min_khz=read(f'{b}/cpufreq/scaling_min_freq'), max_khz=read(f'{b}/cpufreq/scaling_max_freq'),
                       cur_khz=read(f'{b}/cpufreq/scaling_cur_freq'), max_capacity=read(f'{b}/cpu_capacity'))
    cores = []
    for c in cpus:
        sib = info[c]['siblings']
        if sib not in [x['siblings'] for x in cores]:
            cores.append(dict(siblings=sib, first=c))
    out = dict(cpus=info, physical_cores=cores, affinity_now=sorted(os.sched_getaffinity(0)),
               no_turbo=read('/sys/devices/system/cpu/intel_pstate/no_turbo'), boost=read('/sys/devices/system/cpu/cpufreq/boost'),
               suggested_profiles={'P1_cpu2': [cores[1]['first']]} if len(cores) > 1 else {})
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
