using FastHankelTransform, FastGaussQuadrature, BenchmarkTools, Plots, Plots.Measures, LaTeXStrings, Printf, DelimitedFiles, Random, LinearAlgebra

include("/Users/beckman/.julia/config/custom_colors.jl")

gr(size=(350,300))
default(margins=1mm, fontfamily="Computer Modern", label="")
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)

ns  = round.(Int64, 10 .^ range(3, 7, 10))
ps  = round.(Int64, 10 .^ range(5, 9, 10))

ns_both = 2 .^ (8:2:20)

function test_case(case, ns, ps, i; nu=0)
    if case == :exp
        n  = ns[i]
        m  = n
        ws = 10 .^ range(-log10(n)/2, log10(n)/2, n)
        cs = randn(n)
        rs = ws
        p  = (ws[end] - ws[1])*(rs[end] - rs[1])
        return rs, cs, ws, n, m, p
    elseif case == :roots
        n  = ns[i]
        m  = n
        ws = FastGaussQuadrature.approx_besselroots(nu, n+1)
        cs = randn(n)
        rs = ws[1:end-1] / ws[end]
        ws = ws[1:end-1]
        p  = (ws[end] - ws[1])*(rs[end] - rs[1])
        return rs, cs, ws, n, m, p
    end

    p = ps[1]
    if case == :p
        n = ns[1]
        m = ns[1]
        p = ps[i]
    elseif case == :n
        n = ns[i]
        m = ns[1]
    elseif case == :m 
        n = ns[1]
        m = ns[i]
    end
    rs = collect(range(0, sqrt(ps[1]), n))
    cs = randn(n)
    ws = collect(range(0, p/sqrt(ps[1]), m))

    return rs, cs, ws, n, m, p
end

function dir(nu, rs, cs, ws)
    gs = zeros(Float64, length(ws))
    FastHankelTransform.add_dir!(gs, nu, rs, cs, ws)
    return gs
end

## Test scaling with n, m, and p

n_timings = fill(NaN, 2, length(ns))
m_timings = fill(NaN, 2, length(ns))
p_timings = fill(NaN, 2, length(ps))
for (case, timings) in zip([:n, :m, :p], [n_timings, m_timings, p_timings])
    for i in axes(timings, 2)
        rs, cs, ws, n, m, p = test_case(case, ns, ps, i)

        @printf(
            "Timing %s scaling with n = %i, m = %i, p = %.1e (%i of %i)\n", 
            string(case), n, m, p, i, size(timings, 2)
            )
            
        if max(n, m) < 200_000
            timings[1, i] = @belapsed dir(0, $rs, $cs, $ws)
        end

        timings[2, i] = @belapsed nufht(0, $rs, $cs, $ws, tol=1e-8)
    end
end

## Test scaling for scaled roots and exponential

nus  = [0, 10, 20, 50, 100]
tols = [1e-4, 1e-8, 1e-12, 1e-15]
roots_timings = fill(NaN, 2, length(nus), length(tols), length(ns_both))
exp_timings   = fill(NaN, 2, length(ns_both))
for (case, nu, tol, timings) in [
        (:exp, 0, 1e-8, exp_timings),
        [(:roots, 0, tols[j], @view(roots_timings[:,1,j,:])) for j=1:length(tols)]...,
        [(:roots, nus[j], 1e-8, @view(roots_timings[:,j,2,:])) for j=2:length(nus)]...
    ]
    for i in axes(timings, 2)
        rs, cs, ws, n, m, p = test_case(case, ns_both, nothing, i, nu=nu)

        @printf(
            "Timing %s scaling with ν = %i, ε = %.0e, n = %i, p = %.1e (%i of %i)\n", 
            string(case), nu, tol, n, p, i, size(timings, 2)
            )

        if max(n, m) < 100_000 && tol == 1e-8
            timings[1, i] = @belapsed dir($nu, $rs, $cs, $ws)
        end

        FastHankelTransform.setup_nufht!(nu, tol)

        timings[2, i] = @belapsed nufht($nu, $rs, $cs, $ws, tol=$tol)
    end
end

## Test accuracy as a function of tolerance

nu = 0
ns_acc = 10 .^ [3, 5, 7]
tols = 10.0 .^ (-4:-1:-15)
gs_true  = [fill(NaN, n) for n in ns_acc]
gs_nufht = [fill(NaN, length(tols), n) for n in ns_acc]

for (j, n) in enumerate(ns_acc)
    # make coefficient vector with small number of nonzeros
    nnz = minimum(ns_acc)
    I = Random.randperm(n)[1:nnz]
    V = randn(nnz)
    cs = zeros(n)
    cs[I] .= V
    rs, _, ws = test_case(:roots, [n], nothing, 1, nu=nu)

    @printf(
            "Computing direct sum with ν = %i, n = %i\n", 
            nu, n
            )
    gs_true[j] .= dir(nu, rs[I], cs[I], ws)

    for (i, tol) in enumerate(tols)
        @printf(
            "Computing NUFHT with ν = %i, ε = %.0e, n = %i (%i of %i)\n", 
            nu, tol, n, i, length(tols)
            )

        FastHankelTransform.setup_nufht!(nu, tol)

        gs_nufht[j][i, :] = nufht(nu, rs, cs, ws, tol=tol)
    end
end

## Plot accuracy as a function of tolerance

max_errs = [maximum(abs.(gs_true[j] - gs_nufht[j][i,:])) / maximum(abs.(gs_true[j])) for j in eachindex(ns_acc), i in eachindex(tols)]

l2_errs = [norm(gs_true[j] - gs_nufht[j][i,:]) / norm(gs_true[j]) for j in eachindex(ns_acc), i in eachindex(tols)]

pl = plot(
    tols, l2_errs',
    line=(2, scrungle'), marker=(3, scrungle'), markerstrokewidth=0,
    xlabel=L"$\varepsilon$", ylabel=L"rel. $\ell^2$ error",
    label=reshape([latexstring(@sprintf("\$n=10^{%i}\$", Int(log10(n)))) for n in ns_acc], 1, :),
    xscale=:log10, yscale=:log10, 
    ylims=(5e-16, 5e-4), xlims=(5e-16, 5e-4),
    legend=:bottomright
    )
plot!([1e-16, 1e-2], [1e-16, 1e-2], line=(1, :black, :dash))
display(pl)
savefig(pl, "~/Downloads/accuracy.pdf")

##

function plot_scaling(xs, ys, refs; xlabel="", ylims=[1e-2, 1e2], label="")
    pl = plot(
        xs, ys, line=(2, [scrungle[1] scrungle[2]]),
        label=label,
        marker=(3, [scrungle[1] scrungle[2]]), markerstrokewidth=0,
        xlabel=xlabel, ylabel="time (s)",
        xscale=:log10, yscale=:log10, 
        ylims=ylims, legend=:bottomright
        )
    plot!(pl, xs, refs, line=(2, :black, :dash))

    return pl
end

cn = 0.8 * n_timings[1,end] / ns[end]
n_refs = cn * ns
pn = plot_scaling(
    ns, n_timings', n_refs, 
    xlabel=L"$n$", ylims=[1e-2, 1e1], label=["NUFHT" "Direct"]
    )
display(pn)
savefig(pn, "~/Downloads/n_scaling.pdf")

cm = 0.8 * m_timings[1,end] / ns[end]
m_refs = cm * ns
pm = plot_scaling(ns, m_timings', m_refs, xlabel=L"$m$", ylims=[1e-2, 1e1])
display(pm)
savefig(pm, "~/Downloads/m_scaling.pdf")

cp = 0.4 * p_timings[1,end] / (ps[end]*log(ps[end]))
p_refs = cp * ps .* log.(ps)
pp = plot_scaling(ps, p_timings', p_refs, xlabel=L"$p$", ylims=[1e-2, 1e2])
savefig(pp, "~/Downloads/p_scaling.pdf")

##

pl = plot(
    ns_both, hcat(roots_timings[1,:], exp_timings[1,:], exp_timings[2,:]), 
    line=(2, scrungle[[2,3,1]]'),
    label=["NUFHT roots" "NUFHT exp" "Direct"],
    marker=(3, scrungle[[2,3,1]]'), markerstrokewidth=0,
    xlabel=L"$n$", ylabel="time (s)",
    xscale=:log10, yscale=:log10, 
    ylims=[1e-3, 1e3], legend=:bottomright
    )
# n_refs = 0.7 * roots_timings[1,end] / ns_both[end] * ns_both
# plot!(pl, ns_both[5:end], n_refs[5:end], line=(2, :black, :dash))

nlogn_refs = 0.3 * exp_timings[1,end] / (ns_both[end]*log(ns_both[end])) * ns_both .* log.(ns_both)
plot!(pl, ns_both[4:end], nlogn_refs[4:end], line=(2, :black, :dash))

n2_refs = 1.8 * exp_timings[2,4] / (ns_both[4]^2) * ns_both .^ 2
plot!(pl, ns_both[3:5], n2_refs[3:5], line=(2, :black, :dashdot))

savefig(pl, "~/Downloads/both_scaling.pdf")

##

colors = palette(["#FF8A8E", "#8AFFFB"], length(nus))[1:end]'

pl = plot(
    ns_both, roots_timings[2, :, 2, :]', 
    line=(2, colors),
    label=reshape([latexstring(@sprintf("\$\\nu=%i\$", nu)) for nu in nus], 1, :),
    marker=(3, colors), markerstrokewidth=0,
    xlabel=L"$n$", ylabel="time (s)",
    xscale=:log10, yscale=:log10, 
    ylims=[1e-3, 1e3], legend=:topleft
)
plot!(pl, 
    ns_both, roots_timings[1, 1, 2, :], 
    line=(2, :black), marker=(3, :black), label="Direct"
    )
display(pl)
savefig(pl, "~/Downloads/nu_scaling.pdf")

##

colors = reshape(["#FFD1D3", palette(["#FF8A8E", "#660003"], length(tols)-1)[1:end]...], 1, :)

pl = plot(
    ns_both, roots_timings[2, 1, :, :]', 
    line=(2, colors),
    label=reshape([latexstring(@sprintf("\$\\varepsilon=10^{%i}\$", log10(tol))) for tol in tols], 1, :),
    marker=(3, colors), markerstrokewidth=0,
    xlabel=L"$n$", ylabel="time (s)",
    xscale=:log10, yscale=:log10, 
    ylims=[1e-3, 1e3], legend=:topleft
)
plot!(pl, 
    ns_both, roots_timings[1, 1, 2, :], 
    line=(2, :black), marker=(3, :black), label="Direct"
    )
display(pl)
savefig(pl, "~/Downloads/tol_scaling.pdf")