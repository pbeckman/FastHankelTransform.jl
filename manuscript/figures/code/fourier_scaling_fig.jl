using LinearAlgebra, FastGaussQuadrature, FINUFFT, FastHankelTransform
using Printf, Plots, TimerOutputs, BenchmarkTools, LaTeXStrings, JLD
import Bessels: besselj

include("../../../example/util.jl")
include("/Users/beckman/.julia/config/custom_colors.jl")

function quad1D(m, a, b)
    # Gauss-Legendre on [a,b]
    rs, wts = gausslegendre(m)
    rs   .= (rs .+ 1) * (b-a)/2 .+ a
    wts .*= (b-a)/2

    return rs, wts
end

function quad2D(mgl, mtrs, a, b)
    # Gauss-Legendre cross trapezoidal on the annulus [a,b]
    rsgl, wtsgl = quad1D(mgl, a, b)

    rs  = Matrix{Float64}(undef, 2, sum(mtrs))
    wts = Vector{Float64}(undef, sum(mtrs))
    i = 1
    for (j, mtr) in enumerate(mtrs)
        rs[1,i:(i+mtr-1)] .= rsgl[j]
        rs[2,i:(i+mtr-1)] .= range(0, stop=2pi, length=mtr+1)[1:end-1]
        wts[i:(i+mtr-1)]  .= wtsgl[j] * 2pi/mtr

        i += mtr
    end

    return rs, wts
end

function hankel_integrate2D(f, rs, wts, ws; tol=1e-8)
    # use the Hankel transform to compute 2D Fourier transform
    return 2pi * nufht(
        0, 
        rs, f.(rs) .* rs .* wts, 
        ws, tol=tol
        )
end

function fourier_integrate1D(r, mtr, ws; tol=1e-8)
    # use the 1D Fourier transform to compute marginal Fourier transform 
    # on a ring of fixed radius r
    ss = range(0, stop=2pi, length=mtr+1)[1:end-1]
    return real.(nufft1d3(
        r*cos.(ss),
        2pi/mtr * ones(ComplexF64, mtr), 
        -1, tol,
        ws
        ))
end

function fourier_integrate2D(f, rs, wts, ws; tol=1e-8)
    return real.(nufft2d3(
        rs[1,:] .* cos.(rs[2,:]), zeros(size(rs, 2)),
        Complex.(f.(rs[1,:]) .* rs[1,:] .* wts), 
        -1, tol,
        ws, zeros(length(ws))
        ))
end

##

# choose tolerance
tol = 1e-8

# stop NUFFT evaluation before FINUFFT crashes Julia
max_quad = 10_000_000

# hat function and true Fourier transform
a, b     = 0, 1
f(r)     = 1
f_hat(w) = 2pi * (b * besselj(1, b*w) - a * besselj(1, a*w)) / w

# max frequencies to evaluate
# w_maxs = 2 .^ (6:3:21)
w_maxs = 2 .^ (18:3:21)

# number of points to evaluate
ns = round.(Int64, 10 .^ range(3, stop=7, length=8))

ntest = 1000
rs1, wts1 = [], []
m1s = zeros(Int64, length(w_maxs))
m2s = zeros(Int64, length(w_maxs))
m2trss = Vector{Vector{Int64}}(undef, length(w_maxs))
time_1Ds = fill(NaN, length(w_maxs), length(ns))
time_2Ds = fill(NaN, length(w_maxs), length(ns))
aborted_1D = false
aborted_2D = false
for (j, w_max) in enumerate(w_maxs)
    # use coarsest grid
    n  = ns[1]
    ws = collect(range(0, stop=w_max, length=n+1))[2:end]
    f_hat_true = f_hat.(ws)

    if !aborted_1D
        # adaptively determine necessary quadrature
        m1 = 8
        converged = false
        f_hat_1D  = zeros(Float64, n)
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
        m2trs = Vector{Int64}(undef, m2gl)
        f_hat_2D = zeros(Float64, n)
        for k=1:m2gl
            if k == 1 || mod(k, ceil(Int64, m2gl / ntest)) == 0 
                m2tr = k == 1 ? 4 : div(m2trs[k-1],2)
                converged  = false
                f_hat_ref  = fourier_integrate1D(rs1[k], m2tr, ws; tol=tol)
                while !converged
                    m2tr *= 2
                    if m2tr > max_quad 
                        @warn "marginal trapezoidal quadrature with $m2tr > $max_quad = max_quad nodes!"
                        aborted_2D = true
                        break
                    end
                    f_hat_circle = fourier_integrate1D(rs1[k], m2tr, ws; tol=tol)
                    rel_err = norm(f_hat_ref .- f_hat_circle) / norm(f_hat_ref)
                    f_hat_ref .= f_hat_circle
                    @printf("marginal trapezoidal quadrature with %i nodes on circle of radius %.3f gives relative error %.2e\n", m2tr, rs1[k], rel_err)
                    converged = (rel_err < tol*100)
                end
                m2trs[k] = m2tr
            else
                m2trs[k] = m2trs[k-1]
            end
        end
        if sum(m2trs) > max_quad 
            @warn "2D quadrature with $(sum(m2trs)) > $max_quad = max_quad nodes!"
            aborted_2D = true
        else
            rs2, wts2 = quad2D(m2gl, m2trs, a, b)
            f_hat_2D = fourier_integrate2D(f, rs2, wts2, ws, tol=tol)
            rel_err  = norm(f_hat_true .- f_hat_2D) / norm(f_hat_true)
            @printf("2D quadrature with %i nodes gives relative error %.2e\n", sum(m2trs), rel_err)
        end
        m2s[j]    = sum(m2trs)
        m2trss[j] = m2trs
    end
    
    for (i, n) in enumerate(ns)
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

        save(
            "./fourier_scaling.jld", 
            "w_maxs", w_maxs,
            "ns", ns,
            "time_1Ds", time_1Ds,
            "time_2Ds", time_2Ds
        )

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

        save(
            "./fourier_scaling.jld", 
            "w_maxs", w_maxs,
            "ns", ns,
            "time_1Ds", time_1Ds,
            "time_2Ds", time_2Ds
        )

        @printf("
--------------------------------------------------------------
n = %i, tol = %.1e, w_max = %.2e
2D quadrature with %i nodes: %.2e sec
--------------------------------------------------------------\n", 
        n, tol, w_max, sum(m2trs), time_2Ds[j, i]
        )
    end
end

## Plot example 2D quadrature

gr(size=(300,300))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)

j = 1
rs2, _ = quad2D(m1s[j], m2trss[j], a, b)
rs1, _ = quad1D(m1s[j], a, b)

nplot = 300
ts = range(-b, b, nplot)
# heatmap(ts, ts, ones(nplot,1) * cos.(w_maxs[j]*ts'), color=:acton, alpha=0.2, legend=:none)
pl = scatter(
    rs2[1,:].*cos.(rs2[2,:]), rs2[1,:].*sin.(rs2[2,:]), 
    marker=(1.2, :black), label="",
    xlabel=L"$x_1$", ylabel=L"$x_2$"
    )
scatter!(pl, 
rs1, zeros(m1s[j]), 
    marker=(2.5, scrungle[10]), label="",
    markerstrokewidth=1, markerstrokecolor=:white
    )

display(pl)
savefig(pl, "~/Downloads/quadrature_2D.pdf")

## Plot scaling

if isfile("./fourier_scaling.jld")
    dict     = load("./fourier_scaling.jld")
    ns       = dict["ns"]
    w_maxs   = dict["w_maxs"]
    time_1Ds = dict["time_1Ds"]
    time_2Ds = dict["time_2Ds"]
end

gr(size=(400,300))

pal = scrungle

markers = [:circle, :rect, :utriangle, :cross, :star5, :diamond, :hexagon,  :xcross,  :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star6, :star7, :star8, :vline, :hline, :+, :x]

pl = plot(
        xscale=:log10, yscale=:log10,
        xlabel="n", ylabel="time (s)",
        labels=["1D Hankel" "2D Fourier"],
        legend=:topleft, legend_columns=2, 
        legendtitle=" ", legendtitlefontsize=10,
        ylims=(1e-3, 2e2),
        yticks=([1e-2, 1e0, 1e2], [L"10^{-2}", L"10^0", L"10^{2}"]),
        yformatter=:scientific
        )
for j in eachindex(w_maxs[1:4])
    plot!(pl,
        ns, time_1Ds[j, :], 
        marker=(4, markers[j]), markerstrokewidth=0, 
        color=pal[j], line=(2, :solid),
        label=string(Int64(log2(w_maxs[j])))
        # label=latexstring(@sprintf("\\log_2\\omega_{max} = %i", log2(w_maxs[j])))
    )
    plot!(pl,
        ns, time_2Ds[j, :], 
        marker=(4, markers[j]), markerstrokewidth=0, 
        color=pal[j], line=(2, :dash),
        label=""
    )
end
display(pl)
savefig(pl, "~/Downloads/fourier_scaling.pdf")

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