"""Fixed-budget PGD engines; CUDA Graph changes execution, not the objective."""
import hashlib


def flat(obj):
    import torch
    if torch.is_tensor(obj):yield obj
    elif isinstance(obj,(tuple,list)):
        for value in obj:yield from flat(value)
    elif isinstance(obj,dict):
        for value in obj.values():yield from flat(value)


def prediction(raw,mode):
    if mode=='native':return next(t for t in flat(raw) if t.dim()==3 and tuple(t.shape[1:])==(300,38))
    return next(t for t in flat(raw) if t.dim()==3 and t.shape[1]==116)


def objective(raw,mode):
    import torch
    p=prediction(raw,mode)
    scores=p[:,:,4] if mode=='native' else p[:,4:84].amax(1)
    return torch.sigmoid((scores-.25)/.02).sum()


def initial_delta(images,names,seed,model):
    import torch
    delta=torch.zeros_like(images)
    if seed==0:return delta
    for j,name in enumerate(names):
        key=f'{seed}|{model}|{name}'.encode()
        integer=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')%(2**63-1)
        generator=torch.Generator(device=images.device).manual_seed(integer)
        delta[j].uniform_(-16/255,16/255,generator=generator)
    return (images+delta).clamp(0,1)-images


def eager(net,images,delta,mode,steps=100):
    import torch
    d=delta.detach().clone().requires_grad_(True)
    for _ in range(steps):
        loss=objective(net((images+d).clamp(0,1)),mode)
        loss.backward()
        with torch.no_grad():
            d.add_((2/255)*d.grad.sign()).clamp_(-16/255,16/255)
            d.copy_((images+d).clamp(0,1)-images)
            d.grad.zero_()
    return (images+d).clamp(0,1).detach()


class GraphedPGD:
    def __init__(self,net,example,mode):
        import torch
        self.image=example.detach().clone()
        self.delta=torch.zeros_like(example,requires_grad=True)
        self.graph=torch.cuda.CUDAGraph()
        def step():
            loss=objective(net((self.image+self.delta).clamp(0,1)),mode)
            loss.backward()
            with torch.no_grad():
                self.delta.add_((2/255)*self.delta.grad.sign()).clamp_(-16/255,16/255)
                self.delta.copy_((self.image+self.delta).clamp(0,1)-self.image)
                self.delta.grad.zero_()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(8):step()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        with torch.cuda.graph(self.graph,stream=stream):step()

    def __call__(self,images,delta,steps=100):
        import torch
        with torch.no_grad():
            self.image.copy_(images);self.delta.copy_(delta);self.delta.grad.zero_()
        for _ in range(steps):self.graph.replay()
        return (self.image+self.delta).clamp(0,1).detach()
