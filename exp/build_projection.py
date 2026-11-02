"""원본 prepare_deep_projection_gpu1.py 의 수식을 그대로, bf16 2GPU 샤딩용으로 옮긴 것.

원본과 동일: uncentered second moment의 null space, thin SVD, threshold 0.02,
protected = Vh[eig >= thr].T, projection = I - P P^T, residual 검증.
원본과 다름: 모델 dtype(bf16), 2GPU 샤딩, 키 캡처를 CPU float32로 모음.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import common


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="13,14,15")
    p.add_argument("--threshold", type=float, default=0.02)
    a = p.parse_args()

    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    layers = [int(x) for x in a.layers.split(",")]
    raw = Path(a.corpus).read_bytes()
    records = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    texts = [x if isinstance(x, str) else x["text"] for x in records]
    print(f"corpus texts: {len(texts)}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    tok, model = common.load_model()
    keys = {layer: [] for layer in layers}
    handles = []
    for layer in layers:
        def capture(module, args, layer=layer):
            keys[layer].append(args[0][0].detach().float().cpu())
        handles.append(model.get_submodule(f"model.layers.{layer}.mlp.down_proj")
                      .register_forward_pre_hook(capture))
    try:
        for i, text in enumerate(texts):
            x = tok(text, return_tensors="pt", truncation=False).to("cuda:0")
            with torch.inference_mode():
                model(**x, use_cache=False)
            if (i + 1) % 50 == 0:
                print(f"forward {i + 1}/{len(texts)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    dev = "cuda:0"
    projections, diagnostics = [], []
    for layer in layers:
        matrix = torch.cat(keys[layer]).to(dev)
        n, d = matrix.shape
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        eigenvalues = singular.square() / n
        protected = vh[eigenvalues >= a.threshold].T
        protected = torch.linalg.qr(protected, mode="reduced").Q
        projection = torch.eye(d, device=dev) - protected @ protected.T
        residual = (projection @ protected).norm().item()
        diagnostics.append(dict(layer=layer, texts=len(texts), tokens=n,
                                protected_rank=protected.shape[1],
                                null_dimension=d - protected.shape[1],
                                protected_residual_norm=residual))
        print("PROJECTION", diagnostics[-1], flush=True)
        if residual > 1e-3:
            raise ValueError("Projector residual too large")
        projections.append(projection.cpu())
        del matrix, singular, vh, protected, projection
        torch.cuda.empty_cache()

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack(projections), out)
    out.with_suffix(".json").write_text(json.dumps(dict(
        layers=layers, corpus=a.corpus,
        corpus_sha256=hashlib.sha256(raw).hexdigest(), threshold=a.threshold,
        method="Null space of empirical uncentered token second moment via thin SVD",
        note="원본과 같은 수식, bf16/2GPU 로컬 실행", diagnostics=diagnostics), indent=2))
    print("COMPLETED projection", out, flush=True)


if __name__ == "__main__":
    main()
