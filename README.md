# FastHankelTransform.jl

This package uses local and asymptotic expansions of the Bessel function $J_\nu$
to compute the nonuniform fast Hankel transform (NUFHT)
```math
    g_j = \sum_{k=1}^n c_k \, J_\nu(\omega_j r_k)
   \qquad \text{for } j = 1, \ldots, m
```
to a user-specified tolerance $\varepsilon$. The frequencies $\omega_j$ and
source locations $r_k$ can be arbitrarily chosen and need not lie on a grid. The
computational cost of our approach is near optimal, scaling quasilinearly with
$n$ and $m$.

A minimal demo is as follows:
```julia
using FastHankelTransform

# order and tolerance
nu  = 0
tol = 1e-12

# frequencies w_k at which to evaluate the transform
ws = 10 .^ range(-2, 2, 100_000)

# source locations r_k and strengths c_k
rs = rand(10_000)
cs = randn(10_000)

# compute NUFHT
gs = nufht(nu, rs, cs, ws; tol=tol)
```
See the `scripts` directory for more detailed, heavily commented demos. 