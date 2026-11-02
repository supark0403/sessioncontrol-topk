"""code/ 를 수정하지 않고 bf16 모델에서 동작시키기 위한 최소 래퍼.

- one_key_factors: projector/key 를 float32 로 계산해 factor 를 float32 로 저장
  (원본과 동일한 식, dtype 만 맞춤)
- factor_scores: 저장된 float32 factor 와 bf16 key 의 dtype 불일치 해소
"""
import torch


def install():
    import compact_session_memory as csm

    def one_key_factors(projector, key, residual, l2):
        if key.ndim != 1 or residual.ndim != 1 or l2 <= 0:
            raise ValueError("Only one key, one residual and positive L2 supported")
        p = projector.float()
        k = key.float().reshape(-1)
        r = residual.float().reshape(-1)
        projected = p @ k
        denominator = l2 + k @ projected
        if not torch.isfinite(denominator) or denominator.item() <= 0:
            raise ValueError("Invalid Sherman-Morrison denominator")
        return r.detach(), (projected / denominator).detach()

    def factor_scores(payload, keys):
        sample = next(iter(keys.values()))
        numerator = torch.zeros(sample.shape[0], dtype=torch.float64, device=sample.device)
        denominator = torch.zeros((), dtype=torch.float64, device=sample.device)
        with torch.inference_mode():
            for name, (left, right) in payload.items():
                l = left.to(numerator.device).double()
                r = right.to(numerator.device).double()
                key = keys[name].to(numerator.device).double()
                lnorm = l.square().sum()
                rnorm = r.square().sum()
                numerator += (key @ r).square() * lnorm
                denominator += lnorm * rnorm
        if denominator.item() <= 0:
            raise ValueError("Empty compact delta")
        return (numerator / denominator).cpu().tolist()

    csm.one_key_factors = one_key_factors
    csm.factor_scores = factor_scores
