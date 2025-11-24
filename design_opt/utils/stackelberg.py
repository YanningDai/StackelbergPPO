import torch
from torch import nn
from typing import List, Tuple, Callable, Dict, Any, Optional



def _flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
    flat = [t.reshape(-1) for t in tensors if t is not None]
    if len(flat) == 0:
        return torch.tensor([], dtype=torch.float32, device=tensors[0].device)
    return torch.cat(flat)


def _unflatten_like(flat_vec: torch.Tensor, like_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    out = []
    offset = 0
    for t in like_tensors:
        numel = t.numel()
        chunk = flat_vec[offset: offset + numel].view_as(t)
        out.append(chunk)
        offset += numel
    return out

def _fill_none(grads, params):
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

def fisher_vector_product_selfkl(
    policy_exec,          # callable that returns a distribution for the execution (θ2) branch or a policy head containing only execution (θ2)
    theta2_params,        # list of θ2 parameters
    states,               # [B, ...] states only
    v_flat,               # flattened vector matching theta2's dimensions
    damping=1e-2,
    J_c=None, 
    fisher_correct=False,
):
    """
    Fv ≈ ∇_θ2^2 E_s[ KL(π_θ2(·|s) || stopgrad(π_θ2(·|s))) ] · v + λ v
    Does not depend on actions; does not require the old policy; numerically more stable.
    """
    # KL_self
    dist_new = policy_exec(states)                 
    with torch.no_grad():
        dist_ref = policy_exec(states)             
        
    kl_i = torch.distributions.kl.kl_divergence(dist_new, dist_ref)
    if kl_i.dim() > 1:
        kl_i = kl_i.sum(dim=-1)             # e.g., sum per-dimension KL for DiagGaussian -> [B]
    kl = kl_i.mean()
    
    # φ = KL_self (+ α·J_c)
    if fisher_correct:
        assert J_c is not None, "When fisher_correct=True, J_c (a scalar depending on θ2) must be provided"
        phi = kl - J_c
    else:
        phi = kl # objective; note sign conventions (loss may need negation)
    
    # g = ∇θ2 φ
    g_list = torch.autograd.grad(phi, theta2_params, create_graph=True, retain_graph=True, allow_unused=True)
    g_list = _fill_none(g_list, theta2_params)
    g_flat = torch.cat([g.reshape(-1) for g in g_list])

    # 4) HVP = ∇θ2 (g·v)
    dot_gv = (g_flat * v_flat).sum()
    hvp_list = torch.autograd.grad(dot_gv, theta2_params, retain_graph=True, allow_unused=True)
    hvp_list = _fill_none(hvp_list, theta2_params)
    hvp_flat = torch.cat([h.reshape(-1) for h in hvp_list])

    return hvp_flat + damping * v_flat

def conjugate_gradient(fvp_fn, b, max_iter=20, tol=1e-4, tol_abs=1e-8, verbose = False):
    """
    Conjugate Gradient solver for (F+λI) x = b. fvp_fn(x) should return (F+λI)x.
    Prints ||r|| and the relative residual (relative to r0) each iteration.
    """
    x = torch.zeros_like(b)
    r = b.clone()            # x0=0 -> r0 = b - A*0 = b
    p = r.clone()
    rTr = torch.dot(r, r)
    r0_norm = r.norm() + 1e-12

    if verbose:
        print(f"[CG] init  ||r||={r0_norm.item():.3e}")

    for k in range(max_iter):
        Ap = fvp_fn(p)
        pAp = torch.dot(p, Ap)
        if verbose:
            print(f"[Sanity] p^T A p = {pAp:.3e}")  # expected > 0
        
        if pAp <= 0:
            print(f"[CG] k={k:02d}  pAp={pAp.item():.3e} (A not SPD or numeric issue) — stop")
            break

        alpha = rTr / (pAp + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap

        rTr_new = torch.dot(r, r)
        r_norm = r.norm().item()
        rel = (r_norm / r0_norm.item())

        if verbose:
            print(f"[CG] k={k:02d}  alpha={alpha.item():.3e}  ||r||={r_norm:.3e}  rel={rel:.3e}")
                
        if (rel < tol) or (r_norm < tol_abs):
            if verbose:
                print(f"[CG] converged at k={k:02d}  rel={rel:.3e} < tol={tol:.1e}, or r_norm={r_norm:.3e} < tol_abs={tol_abs:.1e}")
            break

        beta = rTr_new / (rTr + 1e-12)
        if not torch.isfinite(beta):
            if verbose:
                print(f"[CG] k={k:02d}  beta not finite — stop")
            break
        
        p = r + beta * p
        rTr = rTr_new

    return x

# Implicit gradient computation for Stackelberg bilevel optimization
def bilevel_leader_grad_correct(
    J_a: torch.Tensor,                      # scalar: depends on theta2 (used for s2) and retains computation graph
    J_c: torch.Tensor,                      # scalar: depends on theta2 (v)
    theta1_list: List[torch.Tensor],
    theta2_list: List[torch.Tensor],
    policy_exec,
    F_states: torch.Tensor,                          # detached states used to compute Fisher/Hessian-vector products
    damping: float = 1e-2,
    cg_max_iter=20,
    cg_tol=5e-4,
    verbose: bool = False,
    fisher_correct: bool = False,
):
    """
    Returns: (∇_{θ1} J1 - K^T s).
    Here s solves (H + damping*I) s = v.
    """

    # ---- v = g_c = ∇_{θ2} J_c ----                    
    v_list = torch.autograd.grad(J_c, theta2_list, create_graph=False, allow_unused=True, retain_graph=True)
    v_list = _fill_none(v_list, theta2_list)
    v_flat = _flatten(v_list).detach()
    
    # ---- s = (F + λI)^(-1) v ----
    def fvp(x: torch.Tensor) -> torch.Tensor:
        return fisher_vector_product_selfkl(policy_exec=policy_exec, theta2_params=theta2_list, states=F_states, v_flat=x, damping=damping, J_c=J_c, fisher_correct=fisher_correct)
    
    s_flat = conjugate_gradient(fvp, v_flat, max_iter=cg_max_iter, tol=cg_tol, verbose = verbose).detach()
    
    # ---- g_a = ∇_{θ2} J_a ----
    g_a_list = torch.autograd.grad(J_a, theta2_list, create_graph=True, allow_unused=True, retain_graph=True)
    g_a_list = _fill_none(g_a_list, theta2_list)
    g_a_flat = _flatten(g_a_list)
    s_2 = 1 * (g_a_flat * s_flat).sum()

    if verbose:
        B = len(F_states)
        print(f"[FVP+CG] θ1={sum(p.numel() for p in theta1_list)}, "
            f"θ2={sum(p.numel() for p in theta2_list)}, B={B}, "
            f"λ={float(damping)}, cg_it={cg_max_iter}")
    
    return s_2


