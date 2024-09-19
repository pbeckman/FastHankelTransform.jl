using LinearAlgebra, FastGaussQuadrature, FINUFFT, FastHankelTransform
using Printf, Plots, TimerOutputs, BenchmarkTools, LaTeXStrings
import Bessels: besselj

include("util.jl")
include("/Users/beckman/.julia/config/custom_colors.jl")

function quad1D(m, a, b)
    # Gauss-Legendre on [a,b]
    rs, wts = gausslegendre(m)
    rs   .= (rs .+ 1) * (b-a)/2 .+ a
    wts .*= (b-a)/2

    return rs, wts
end

function quad2D(mgl, mtr, a, b)
    # Gauss-Legendre cross trapezoidal on the annulus [a,b]
    rsgl, wtsgl = quad1D(mgl, a, b)

    rs = hcat(collect.(Iterators.product(
        rsgl, 
        range(0, stop=2pi, length=mtr+1)[1:end-1]
        ))...)
    wts = 2pi/mtr * repeat(wtsgl, mtr)

    return rs, wts
end

function hankel_integrate2D(f, rs, wts, ws; tol=1e-8)
    return 2pi * nufht(
        0, 
        rs, f.(rs) .* rs .* wts, 
        ws, tol=tol
        )
end

function fourier_integrate2D(f, rs, wts, ws; tol=1e-8)
    return real.(nufft2d3(
        rs[1,:] .* cos.(rs[2,:]), zeros(size(rs, 2)),
        Complex.(f.(rs[1,:]) .* rs[1,:] .* wts), 
        +1, tol,
        ws, zeros(length(ws))
        ))
end

# choose tolerance
tol = 1e-12

# stop NUFFT evaluation before FINUFFT crashes Julia
max_quad = 20_000_000

# hat function and true Fourier transform
a, b     = 0, 1
f(r)     = 1
f_hat(w) = 2pi * (b * besselj(1, b*w) - a * besselj(1, a*w)) / w

# max frequencies to evaluate
# w_maxs = 10 .^ (2:6)
w_maxs = 2 .^ (8:4:24)

# number of points to evaluate
ns = round.(Int64, 10 .^ range(3, stop=8, length=10)) 
# ns = round.(Int64, 10 .^ range(3, stop=6, length=3))

m1s = zeros(Int64, length(w_maxs))
m2s = zeros(Int64, length(w_maxs))
time_1Ds = fill(NaN, length(w_maxs), length(ns))
time_2Ds = fill(NaN, length(w_maxs), length(ns))
aborted_1D = false
aborted_2D = false
for (j, w_max) in enumerate(w_maxs)
    # use finest grid
    n  = ns[end]
    ws = collect(range(0, stop=w_max, length=n+1))[2:end]
    f_hat_true = f_hat.(ws)

    if !aborted_1D
        # adaptively determine necessary quadrature
        m1 = 8
        converged  = false
        rs1, wts1 = [], []
        f_hat_1D  = zeros(Float64, n)
        time_1D   = 0.0
        while !converged
            m1 *= 2
            rs1, wts1 = quad1D(m1, a, b)
            if m1 > max_quad 
                @warn "1D quadrature with $m1 > $max_quad = max_quad nodes!"
                aborted_1D = true
                break
            end
            f_hat_1D .= hankel_integrate2D(f, rs1, wts1, ws, tol=tol)
            rel_err   = norm(f_hat_true .- f_hat_1D) / norm(f_hat_true)
            @printf("1D quadrature with %i nodes gives relative error %.2e\n", m1, rel_err)
            converged = (rel_err < tol*100)
        end
        m1s[j] = m1
    end

    if !aborted_2D
        m2gl = m1
        m2tr = 8
        converged  = false
        rs2, wts2 = [], []
        f_hat_2D  = zeros(Float64, n)
        time_2D   = 0.0
        while !converged
            m2tr *= 2
            rs2, wts2 = quad2D(m2gl, m2tr, a, b)
            if m2gl*m2tr > max_quad 
                @warn "2D quadrature with $(m2gl*m2tr) > $max_quad = max_quad nodes!"
                aborted_2D = true
                break
            end
            f_hat_2D .= fourier_integrate2D(f, rs2, wts2, ws, tol=tol)
            rel_err   = norm(f_hat_true .- f_hat_2D) / norm(f_hat_true)
            @printf("2D quadrature with %i nodes gives relative error %.2e\n", m2gl*m2tr, rel_err)
            converged = (rel_err < tol*100)
        end
        m2s[j] = m2gl*m2tr
    end
    
    for (i, n) in enumerate(ns)
        # choose frequencies and evaluate true Fourier transform
        ws = collect(range(0, stop=w_max, length=n+1))[2:end]

        @printf("
==============================================================
w_max = %i (%i of %i), n = %i (%i of %i), tol = %.1e
==============================================================\n", 
        w_max, j, length(w_maxs), n, i, length(ns), tol
        )

        if !aborted_1D
            time_1Ds[j, i] = @belapsed begin 
                hankel_integrate2D($f, $rs1, $wts1, $ws, tol=$tol)
            end
        end

        @printf("
--------------------------------------------------------------
n = %i, tol = %.1e, w_max = %.2e
1D quadrature with %i nodes: %.2e sec
--------------------------------------------------------------\n", 
        n, tol, w_max, m1, time_1Ds[j, i]
        )

        if !aborted_2D
            time_2Ds[j, i] = @belapsed begin 
                fourier_integrate2D($f, $rs2, $wts2, $ws, tol=$tol)
            end
        end

        @printf("
--------------------------------------------------------------
n = %i, tol = %.1e, w_max = %.2e
2D quadrature with %i x %i = %i nodes: %.2e sec
--------------------------------------------------------------\n", 
        n, tol, w_max, m2gl, m2tr, m2gl*m2tr, time_2Ds[j, i]
        )
    end
end

##

gr(size=(600,500))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)

pal = scrungle

markers = [:circle, :rect, :utriangle, :cross, :star5, :diamond, :hexagon,  :xcross,  :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star6, :star7, :star8, :vline, :hline, :+, :x]


pl = plot(
        xscale=:log10, yscale=:log10,
        xlabel="n", ylabel="time (s)",
        labels=["1D Hankel" "2D Fourier"],
        legend=:topleft
        )
for j in eachindex(w_maxs)
    plot!(pl,
        ns, time_1Ds[j, :], 
        marker=(4, markers[j]), markerstrokewidth=0, 
        color=pal[j], line=(2, :solid),
        label=latexstring(@sprintf("\\omega_{max} = %i", w_maxs[j]))
    )
    plot!(pl,
        ns, time_2Ds[j, :], 
        marker=(4, markers[j]), markerstrokewidth=0, 
        color=pal[j], line=(2, :dash),
        label=""
    )
end
# ref_inds = (length(ns)-3):length(ns)
# c1 = 0.5 * minimum(time_1Ds[:,ref_inds[1]]) / (ns[ref_inds[1]])
# c2 = 0.5 * minimum(time_1Ds[:,ref_inds[1]]) / (ns[ref_inds[1]]*log(ns[ref_inds[1]]))
# plot!(pl,
#     ns[ref_inds], c1 * ns[ref_inds],
#     line=(2, :dashdot, :black),
#     label=""
# )
# annotate!(pl, 3.5e6, 1, L"\mathcal{O}(n)")
# plot!(pl,
#     ns[ref_inds], c2 * ns[ref_inds].*log.(ns[ref_inds]),
#     line=(2, :dashdot, :grey60),
#     label=L"\mathcal{O}(n\log n)"
# )
display(pl)

##

gr(size=(1000, 1000*m1s[end]/n))
pl = skeleton_plot(rs1, ws)
display(pl)

gr(size=(500,500))

if m2gl*m2tr < 100_000
    pl = scatter(
        rs2[1,:] .* cos.(rs2[2,:]), 
        rs2[1,:] .* sin.(rs2[2,:]), 
        marker=(2, :black), xlims=(-b,b), ylims=(-b,b), label=""
        )
    display(pl)
end

sk = max(1, div(n, 1000))

pl = plot(ws[1:sk:end], f_hat_true[1:sk:end], label="true")
plot!(pl, ws[1:sk:end], f_hat_1D[1:sk:end], label="1D quadrature")
plot!(pl, ws[1:sk:end], f_hat_2D[1:sk:end], label="2D quadrature")
display(pl)

pl = plot(yscale=:log10, ylims=(1e-16, 1e0))
plot!(pl,
    ws[1:sk:end], 
    abs.((f_hat_true .- f_hat_1D) ./ f_hat_true)[1:sk:end], 
    label="1D quadrature"
    )
plot!(pl, 
    ws[1:sk:end], 
    abs.((f_hat_true .- f_hat_2D) ./ f_hat_true)[1:sk:end], 
    label="2D quadrature"
    )
display(pl)