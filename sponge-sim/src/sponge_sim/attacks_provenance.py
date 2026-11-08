"""Code COPY từ repo cũ để CHẠY LẠI thuật toán tấn công đã có (sg_ga, ga_center, ga_random_location,
random_search, gradient_tanh = "white-box tanh"). KHÔNG sửa thuật toán. Hai thay đổi được khai rõ dưới
đây là generalize hằng số/thiết bị, không đổi công thức tối ưu.

Dùng cho E3 (tanh_whitebox, mốc "tấn công mạnh"), E4 (sg_ga ở 640px), và làm nguồn patch cho E8.

SOURCES (SHA256 tại thời điểm copy, 21/10/2026):
  experiments/extended_validation.py sha256=5d551d5e3582b449c942a00279c18a026a30b8a5260d4abaf03e9e0dd48da69c
  attack/genetic_algo.py            sha256=9eba0124fd3042b4ec2c470bf82df4ef6deb99cdd747c7d3a0cdd7107057da6b
  core/sponge_fitness.py            sha256=b123a98422a815c69cbdf57287f74f0c73bbde5b2d6869276db44b3b9f4fa9f8

ĐÃ GENERALIZE (không đổi thuật toán, chỉ đổi hằng/thiết bị cố định thành tham số):
  1. find_locations/insert ở bản gốc hardcode canvas 320 (xem experiments/extended_validation.py:62-79:
     '(320-size)//2', 'rng.randrange(321-size)', 'range(0,321-size,step)'). Bản copy nhận `canvas` làm
     tham số (mặc định 320 = hành vi gốc) để chạy đúng ở 640px (E4). KHÔNG đổi thuật toán tìm vị trí.
  2. SpongeGA.__init__ bản gốc tạo self.population trên `cuda if torch.cuda.is_available() else cpu`
     (attack/genetic_algo.py:67,79-82). Trên GPU, torch.rand dùng RNG CUDA (khác RNG CPU) dù cùng số seed
     -> patch khởi tạo khác byte so với CPU, dù CÙNG thuật toán. Ghi rõ trong manifest mọi run, không
     coi đây là lỗi tái lập.
  3. Trên CUDA, VictimModel tự chọn self.model là bản .half() (core/victim_model.py:38-47,
     use_half = cuda_available and device.type=='cuda'). gradient_tanh gọi trực tiếp
     `victim.model(augmented)` với augmented float32 -> lệch dtype trên GPU. sg_ga dùng
     get_raw_predictions, tự ép input sang half nên không crash nhưng đổi độ chính xác so với
     bản CPU gốc (float32), làm lẫn "đổi độ phân giải" với "đổi độ chính xác số".
     Driver script (E3, E4) PHẢI ghi đè hai THUỘC TÍNH INSTANCE (không sửa file) ngay sau khi tạo
     VictimModel: `victim.use_half = False; victim.model = victim.model_float`. model_float là
     thuộc tính đã có sẵn trong VictimModel.__init__; đây là chọn lại thuộc tính có sẵn, không phải
     sửa công thức GA/gradient hay sửa file core/victim_model.py.

  4. Tái lập byte-for-byte yêu cầu torch.set_num_threads(2) (manifest.protocol.torch_threads của X1-500,
     experiments/extended_validation.py:238). gradient_tanh lệch patch đã lưu khi dùng 1 hoặc 12 thread
     (kiểm 21/10/2026: chênh tối đa 14/255); sg_ga khớp. Driver phải đặt và ghi số thread vào manifest.

  4. optimize() nhận thêm spec['threshold'] (mặc định .01 = bản gốc) để truyền cho objective() ở nhánh
     không-gradient; objective() bản gốc đã có tham số threshold, chỉ chưa được truyền (extended_validation.py:92,
     :171-175). Dùng cho thử nghiệm ngưỡng 0,25. Không đổi công thức fitness.
  6. spec['device'] (mặc định 'cpu' = bản gốc, từng bit): đặt tensor ảnh train, mô-đun EOT (kornia) và ứng viên lên
     thiết bị tính toán. Cùng phép tính, khác thiết bị; trên GPU RNG khác RNG CPU nên patch không trùng byte với bản CPU
     (cùng thuật toán). Cần để tối ưu 100 ảnh ở 640 px trong thời gian chấp nhận được.
  5. load_image_canvas(path, canvas): cùng phép resize trực tiếp (BILINEAR, không letterbox) của
     experiments/local_q1_validation.py:64-67 nhưng canvas là tham số (320 = bản gốc, 640 cho thử nghiệm 640 px).

Test hồi quy: tests/test_attacks_provenance.py (budget 400, seed 0, canvas 320) so với
outputs/x1_500img_20261018/optimization/{sg_ga,tanh_whitebox_patch}_seed0/patch_q400.npy.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import cv2
import kornia.augmentation as K
import numpy as np
import torch

MAX_SCORE_DIM = 8400  # anchor count ở 640px (tham chiếu, không dùng để ép logic)


# SOURCE: experiments/local_q1_validation.py:64-67 (load_image) sha256=9739eb66... ; GENERALIZE 5: canvas tham số.
def load_image_canvas(path, canvas=320):
    from PIL import Image
    image = Image.open(path).convert('RGB').resize((canvas, canvas), Image.Resampling.BILINEAR)
    return (torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255).contiguous()


# SOURCE: core/sponge_fitness.py:21-57 (calculate_sponge_fitness) sha256=b123a984...
def calculate_sponge_fitness(batch_scores, conf_thresh=0.01, lambda_weight=1.5):
    active_mask = batch_scores > conf_thresh
    num_boxes_per_image = active_mask.sum(dim=1).float()
    total_conf_per_image = (batch_scores * active_mask).sum(dim=1)
    fitness_scores = total_conf_per_image + lambda_weight * num_boxes_per_image
    return fitness_scores, num_boxes_per_image


# SOURCE: attack/genetic_algo.py:27-91 (SpongeGA.__init__) + 147-168 (_crossover) sha256=9eba0124...
# CHỈ hai phương thức này được optimize() dùng (evolve() của lớp gốc KHÔNG được dùng).
class SpongeGA:
    def __init__(self, patch_size=64, pop_size=20, generations=1, mutation_rate=0.1,
                 mutation_strength=0.2, crossover_mode='horizontal_split', elite_k=5, seed=None,
                 use_saliency=True, convergence_patience=10, convergence_delta=0.5):
        self.patch_size = patch_size
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_mode = crossover_mode
        self.elite_k = min(elite_k, pop_size)
        self.seed = seed
        self.use_saliency = use_saliency
        self.convergence_patience = convergence_patience
        self.convergence_delta = convergence_delta
        # Giữ nguyên hành vi gốc: device tự chọn cuda nếu có. Ghi chú 2 ở trên.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.population = torch.rand((self.pop_size, 3, self.patch_size, self.patch_size),
                                     device=self.device, dtype=torch.float32)

    def _crossover(self, parents1, parents2):
        n = parents1.shape[0]
        if self.crossover_mode == 'horizontal_split':
            split = self.patch_size // 2
            children = parents1.clone()
            children[:, :, split:, :] = parents2[:, :, split:, :]
        elif self.crossover_mode == 'uniform':
            mask = (torch.rand(n, 1, self.patch_size, self.patch_size, device=self.device) > 0.5).float()
            children = mask * parents1 + (1 - mask) * parents2
        elif self.crossover_mode == 'two_point':
            p1 = torch.randint(0, self.patch_size, (1,)).item()
            p2 = torch.randint(p1 + 1, self.patch_size + 1, (1,)).item()
            children = parents1.clone()
            children[:, :, p1:p2, :] = parents2[:, :, p1:p2, :]
        else:
            raise ValueError(f'Unknown crossover_mode: {self.crossover_mode}')
        return children


# SOURCE: experiments/extended_validation.py:41-57 (make_eot) sha256=5d551d5e...
def make_eot(mode, device='cpu'):
    layers = []
    if mode != 'no_rotation':
        layers.append(K.RandomRotation(degrees=15., p=.8))
    if mode != 'no_color':
        layers.append(K.ColorJitter(brightness=.2, contrast=.2, p=.8))
    if mode != 'no_blur':
        layers.append(K.RandomGaussianBlur((3, 3), (.1, 1.), p=.5))
    pipeline = K.AugmentationSequential(*layers, data_keys=['input'], same_on_batch=False).to(device)

    def transform(x):
        if mode == 'none':
            return x
        x = pipeline(x.contiguous())
        if mode != 'no_noise':
            x = x + torch.randn_like(x) * .02
        return x.clamp(0, 1)
    return transform


# SOURCE: experiments/extended_validation.py:60-79 (find_locations) sha256=5d551d5e...
# GENERALIZE 1: thêm tham số canvas (mặc định 320 = hành vi gốc byte-for-byte).
def find_locations(images, size, mode, seed, canvas=320):
    if mode == 'center':
        return [[(canvas - size) // 2] * 2 for _ in images]
    if mode == 'random':
        rng = random.Random(seed + 98171)
        return [[rng.randrange(canvas + 1 - size), rng.randrange(canvas + 1 - size)] for _ in images]
    locations = []
    for image in images:
        gray = cv2.cvtColor((image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        best = (-1., (canvas - size) // 2, (canvas - size) // 2)
        step = max(size // 4, 8)
        coords = sorted(set(range(0, canvas + 1 - size, step)) | {canvas - size})
        for y in coords:
            for x in coords:
                score = cv2.Laplacian(gray[y:y+size, x:x+size], cv2.CV_64F).var()
                if score > best[0]:
                    best = score, y, x
        locations.append([best[1], best[2]])
    return locations


# SOURCE: experiments/extended_validation.py:84-89 (insert) sha256=5d551d5e... (giống provenance.insert)
def insert(images, patches, locations):
    result = images.unsqueeze(0).expand(len(patches), -1, -1, -1, -1).clone()
    size = patches.shape[-1]
    for index, (y, x) in enumerate(locations):
        result[:, index, :, y:y+size, x:x+size] = patches
    return result.flatten(0, 1).contiguous()


# SOURCE: experiments/extended_validation.py:92-95 (objective) sha256=5d551d5e...
def objective(scores, cap=None, threshold=.01):
    return calculate_sponge_fitness(scores, conf_thresh=threshold)[0]


def save_json(path, data):
    import json
    Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding='utf-8')


# SOURCE: experiments/extended_validation.py:113-222 (optimize) sha256=5d551d5e...
# GENERALIZE: find_locations nhận thêm canvas=spec.get('canvas', 320). KHÔNG đổi công thức GA/gradient.
# Driver PHẢI ghi đè victim.use_half/victim.model TRƯỚC khi gọi (xem ghi chú 3 ở đầu file) -- optimize()
# ở đây chỉ dùng `victim.model` (như bản gốc), không tự chọn model_float. Checkpoint đổi tên thành
# patch_q{budget}_c{canvas}.npy khi canvas != 320 để không đè lên artifact 320px nếu dùng cùng thư mục.
def optimize(victim, train_images, spec, seed, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / 'complete.json'
    if completed.exists():
        return __import__('json').loads(completed.read_text(encoding='utf-8'))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    size, draws = spec['size'], spec['draws']
    canvas = spec.get('canvas', 320)
    locations = find_locations(train_images, size, spec['location'], seed, canvas=canvas)
    train = torch.cat(train_images).to(torch.device(spec.get('device', 'cpu')))
    ga = SpongeGA(patch_size=size, pop_size=20, elite_k=5, generations=1, seed=seed)
    dev = torch.device(spec.get('device', 'cpu'))
    transform = make_eot(spec['eot'], dev)
    population = ga.population.cpu()
    patch = population[0].clone()
    best, queries, generation = -float('inf'), 0, 0
    history, checkpoints = [], []
    started = time.perf_counter()
    state_file = run_dir / 'resume.pt'
    resumed_wall = 0.
    if state_file.exists():
        state = torch.load(state_file, map_location='cpu', weights_only=False)
        if state['spec'] != spec:
            raise RuntimeError('Resume specification mismatch')
        population, patch = state['population'], state['patch']
        best, queries, generation = state['best'], state['queries'], state['generation']
        history, checkpoints = state['history'], state['checkpoints']
        resumed_wall = state['wall_seconds']
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        rng.setstate(state['python_rng'])
    device = next(victim.model.parameters()).device
    dtype = next(victim.model.parameters()).dtype

    def checkpoint(budget):
        tag = f'q{budget}_c{canvas}' if canvas != 320 else f'q{budget}'
        np.save(run_dir / f'patch_{tag}.npy', patch.numpy())
        from PIL import Image
        Image.fromarray((patch.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
                        ).save(run_dir / f'patch_{tag}.png')
        import hashlib
        sha = hashlib.sha256(patch.numpy().tobytes()).hexdigest()
        record = dict(seed=seed, specification=spec, queries=budget, best_training_fitness=best,
                      locations=locations, wall_seconds=resumed_wall+time.perf_counter()-started,
                      area_percent=size*size/canvas**2*100, canvas=canvas, patch_sha256=sha)
        save_json(run_dir / f'checkpoint_{tag}.json', record)
        checkpoints.append(record)

    maximum = max(spec['budgets'])
    cost = len(train_images)*draws
    if any(b % cost for b in spec['budgets']):
        raise ValueError('Budgets must be divisible by images times EOT draws')
    while queries < maximum:
        generation += 1
        fits = []
        for index in range(len(population)):
            if queries == maximum:
                break
            candidate = population[index].detach().clone()
            if spec['method'] == 'gradient_tanh':
                candidate = patch.detach().clone().requires_grad_(True)
                augmented = transform(insert(train, candidate.to(dev).unsqueeze(0), locations))
                augmented = augmented.to(device=device, dtype=dtype)  # xem ghi chú 3; không đổi công thức
                pred = victim.model(augmented)
                pred = pred[0] if isinstance(pred, (tuple, list)) else pred
                scores = pred[:, 4:].max(dim=1).values.float()
                smooth = scores.tanh().sum(dim=1).mean()
                grad, = torch.autograd.grad(smooth, candidate)
                with torch.no_grad():
                    current = float(objective(scores).mean())
                    patch = (candidate + grad.sign()/255.).clamp(0, 1).detach()
                evaluated = candidate.detach()
            else:
                with torch.no_grad():
                    values = []
                    for _ in range(draws):
                        x = transform(insert(train, candidate.to(dev).unsqueeze(0), locations))
                        scores = victim.get_raw_predictions(x).float()
                        values.append(objective(scores, spec.get('cap'), spec.get('threshold', .01)).mean())
                    current = float(torch.stack(values).mean())
                evaluated = candidate
            queries += cost
            fits.append(current)
            if spec['method'] != 'gradient_tanh' and current > best:
                best, patch = current, evaluated.clone()
            if spec['method'] == 'gradient_tanh':
                best = current
            history.append(dict(queries=queries, fitness=current, selected_fitness=best))
            if queries in spec['budgets']:
                if spec['method'] == 'gradient_tanh':
                    next_patch = patch
                    patch = evaluated.clone()
                    checkpoint(queries)
                    patch = next_patch
                else:
                    checkpoint(queries)
        if queries < maximum:
            if spec['method'] == 'random_search':
                population = torch.rand_like(population)
            elif spec['method'] != 'gradient_tanh':
                top = torch.tensor(fits).topk(5).indices
                elites = population[top].clone()
                parents1 = elites[torch.randint(0, 5, (15,))]
                parents2 = elites[torch.randint(0, 5, (15,))]
                children = ga._crossover(parents1, parents2)
                mask = (torch.rand(15, 1, 1, 1) < .1)
                children = (children+mask*torch.randn_like(children)*.2).clamp(0, 1)
                population = torch.cat([elites, children])
        temporary = state_file.with_suffix('.tmp')
        torch.save(dict(spec=spec, population=population, patch=patch, best=best,
                        queries=queries, generation=generation, history=history, checkpoints=checkpoints,
                        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=rng.getstate(),
                        wall_seconds=resumed_wall+time.perf_counter()-started), temporary)
        temporary.replace(state_file)
        print(f'{spec["name"]} seed {seed}: {queries}/{maximum} image forwards; score {best:.3f}', flush=True)
    result = dict(seed=seed, spec=spec, checkpoints=checkpoints, history=history,
                  queries=queries, wall_seconds=resumed_wall+time.perf_counter()-started)
    save_json(completed, result)
    return result
