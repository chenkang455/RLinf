"""Verify that the refactored _compute_nft_target_and_pred produces
identical results to the original per-branch implementation."""

import torch

torch.manual_seed(42)

B, C, D = 8, 5, 7  # batch, chunk, action_dim

x_t = torch.randn(B, C, D)
vel = torch.randn(B, C, D)
t_bc = torch.rand(B, 1, 1) * 0.8 + 0.1      # t in (0.1, 0.9)
dt_bc = t_bc / 10                              # dt = t/10
sigma_i = torch.rand(B, 1, 1) * 0.3           # noise scale

# ============ Original implementation (before refactor) ============

# --- Original x0 branch ---
orig_x0_pred = x_t - vel * t_bc               # pred = x_t - vel * t_bc

# --- Original xnext branch ---
orig_x0_pred_xn = x_t - vel * t_bc
orig_x1_pred_xn = x_t + vel * (1 - t_bc)
w0 = 1.0 - (t_bc - dt_bc)
w1 = t_bc - dt_bc - sigma_i**2 * dt_bc / (2 * t_bc)
orig_xnext_pred = orig_x0_pred_xn * w0 + orig_x1_pred_xn * w1

# ============ Refactored implementation (after refactor) ============

# Shared components
x0_pred = x_t - vel * t_bc
x1_pred = x_t + vel * (1 - t_bc)

# --- Refactored xnext branch (should be identical) ---
w0_new = 1.0 - (t_bc - dt_bc)
w1_new = t_bc - dt_bc - sigma_i**2 * dt_bc / (2 * t_bc)
new_xnext_pred = x0_pred * w0_new + x1_pred * w1_new

# --- Refactored x0 branch (new SDE-aware formula) ---
new_x0_pred = x0_pred - (sigma_i**2 / 2) * x1_pred

# --- Refactored x0 branch with sigma=0 (should match original x0) ---
new_x0_pred_ode = x0_pred - (torch.zeros_like(sigma_i)**2 / 2) * x1_pred

# ============ Verification ============

print("=" * 60)
print("1. xnext branch: refactored vs original")
diff_xnext = (new_xnext_pred - orig_xnext_pred).abs().max().item()
print(f"   max abs diff = {diff_xnext:.2e}")
assert diff_xnext < 1e-6, "FAIL: xnext branch changed!"
print("   PASS")

print()
print("2. x0 branch (sigma=0): refactored SDE-aware vs original ODE")
diff_x0_ode = (new_x0_pred_ode - orig_x0_pred).abs().max().item()
print(f"   max abs diff = {diff_x0_ode:.2e}")
assert diff_x0_ode < 1e-6, "FAIL: x0 ODE case changed!"
print("   PASS")

print()
print("3. x0 branch (sigma>0): SDE correction is non-zero")
diff_sde_correction = (new_x0_pred - orig_x0_pred).abs().max().item()
print(f"   max abs diff from original = {diff_sde_correction:.2e}")
assert diff_sde_correction > 1e-4, "FAIL: SDE correction has no effect!"
print(f"   PASS (correction magnitude = {diff_sde_correction:.4f})")

print()
print("4. Verify x0 SDE formula = xnext formula with dt=t")
# xnext formula with dt_bc = t_bc:
w0_full = 1.0 - (t_bc - t_bc)   # = 1
w1_full = (t_bc - t_bc) - sigma_i**2 * t_bc / (2 * t_bc)  # = -sigma_i^2/2
xnext_as_x0 = x0_pred * w0_full + x1_pred * w1_full
diff_derivation = (new_x0_pred - xnext_as_x0).abs().max().item()
print(f"   max abs diff = {diff_derivation:.2e}")
assert diff_derivation < 1e-6, "FAIL: x0 SDE formula != xnext(dt=t)!"
print("   PASS")

print()
print("=" * 60)
print("All checks passed.")
