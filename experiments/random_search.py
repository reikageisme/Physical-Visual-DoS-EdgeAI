import sys, os, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.victim_model import VictimModel
from core.sponge_fitness import calculate_sponge_fitness

victim = VictimModel(device="cpu")
res = 320
patch_size = 64
n_evals = 15 * 20  # 300 evaluations per seed (fair comparison to GA)
n_seeds = 5

best_fitnesses = []
print(f"Running Random Search ({n_seeds} seeds, {n_evals} evals/seed)...")

for seed in range(n_seeds):
    rng = np.random.default_rng(seed)
    best_fit = 0
    base_img = torch.rand((1, 3, res, res), dtype=torch.float32, device="cpu")
    
    for _ in range(n_evals):
        # Random patch [0, 1]
        patch = rng.uniform(0, 1, (1, 3, patch_size, patch_size)).astype(np.float32)
        patch_t = torch.from_numpy(patch).to("cpu")
        
        # Random location
        y = rng.integers(0, res - patch_size)
        x = rng.integers(0, res - patch_size)
        
        img = base_img.clone()
        img[0, :, y:y+patch_size, x:x+patch_size] = patch_t
        
        scores = victim.get_raw_predictions(img)
        fit, _ = calculate_sponge_fitness(scores, conf_thresh=0.01)
        fit = float(fit)
        if fit > best_fit:
            best_fit = fit
            
    best_fitnesses.append(best_fit)
    print(f"  Seed {seed}: {best_fit:.2f}")

mean = np.mean(best_fitnesses)
std = np.std(best_fitnesses)
print("\n" + "="*50)
print(f"Random Search Results: {mean:.2f} ± {std:.2f}")
print("="*50)
