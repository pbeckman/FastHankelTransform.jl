using FastHankelTransform, FastGaussQuadrature, BenchmarkTools, Plots, Plots.Measures, LaTeXStrings, Printf, DelimitedFiles, Random, LinearAlgebra, JLD, Random

Random.seed!(123)

# set up plotting
gr(size=(350,300))
default(margins=1mm, fontfamily="Computer Modern", label="")
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)

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

# test scaling with n, m, and p
if isfile("./nmp_scaling.jld")
    jld = load("./nmp_scaling.jld")
    ns        = jld["ns"]
    ps        = jld["ps"]
    n_timings = jld["n_timings"]
    m_timings = jld["m_timings"]
    p_timings = jld["p_timings"]
else
    ns = round.(Int64, 10 .^ range(3, 8, 10))
    ps = round.(Int64, 10 .^ range(5, 9, 10))
    n_timings = fill(NaN, 2, length(ns))
    m_timings = fill(NaN, 2, length(ns))
    p_timings = fill(NaN, 2, length(ps))
    for (case, timings) in zip([:n, :m, :p], [n_timings, m_timings, p_timings])
        for i in axes(timings, 2)
            local rs, cs, ws, n, m, p = test_case(case, ns, ps, i)

            @printf(
                "Timing %s scaling with n = %i, m = %i, p = %.1e (%i of %i)\n", 
                string(case), n, m, p, i, size(timings, 2)
                )
                
            if max(n, m) < 2_500_000
                timings[1, i] = @belapsed dir(0, $rs, $cs, $ws)
            end

            timings[2, i] = @belapsed nufht(0, $rs, $cs, $ws, tol=1e-8)
        end
    end
    save(
        "./nmp_scaling.jld", 
        "ns", ns,
        "ps", ps,
        "n_timings", n_timings, 
        "m_timings", m_timings, 
        "p_timings", p_timings
        )
end

# plot scaling with n, m, and p 
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

cn = 0.6 * n_timings[2,end] / ns[end]
n_refs = cn * ns
n_refs[1:5] .= NaN
pn = plot_scaling(
    ns, n_timings', n_refs, 
    xlabel=L"$n$", ylims=[1e-2, 1e2], label=["Direct" "NUFHT"]
    )
display(pn)
savefig(pn, "./figures/n_scaling.pdf")

cm = 0.7 * m_timings[2,end] / ns[end]
m_refs = cm * ns
m_refs[1:5] .= NaN
pm = plot_scaling(ns, m_timings', m_refs, xlabel=L"$m$", ylims=[1e-2, 1e2])
display(pm)
savefig(pm, "./figures/m_scaling.pdf")

cp = 0.4 * p_timings[2,end] / (ps[end]*log(ps[end]))
p_refs = cp * ps .* log.(ps)
p_refs[1:5] .= NaN
pp = plot_scaling(ps, p_timings', p_refs, xlabel=L"$p$", ylims=[1e-2, 1e2])
savefig(pp, "./figures/p_scaling.pdf")

# test scaling for m = O(n)
if isfile("./both_scaling.jld")
    jld = load("./both_scaling.jld")
    ns_both       = jld["ns"]
    nus           = jld["nus"]
    tols          = jld["tols"]
    roots_timings = jld["roots_timings"]
    exp_timings   = jld["exp_timings"]
else
    ns_both = 2 .^ (8:2:20)
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

            if max(n, m) < 100_000 && tol == 1e-8 && nu in nus[1:2]
                timings[1, i] = @belapsed dir($nu, $rs, $cs, $ws)
            end

            FastHankelTransform.setup_nufht!(nu, tol)

            timings[2, i] = @belapsed nufht($nu, $rs, $cs, $ws, tol=$tol)
        end
    end
    save(
        "./both_scaling.jld", 
        "ns", ns_both,
        "nus", nus,
        "tols", tols,
        "roots_timings", roots_timings, 
        "exp_timings", exp_timings
        )
end

# plot scaling for m = O(n)
pl = plot(
    ns_both, hcat(roots_timings[2, 1, 2, :], exp_timings[2,:], roots_timings[1, 1, 2, :]), 
    line=(2, scrungle[[2,3,1]]'),
    label=["NUFHT F-B" "NUFHT exp" "Direct"],
    marker=(3, scrungle[[2,3,1]]'), markerstrokewidth=0,
    xlabel=L"$n$", ylabel="time (s)",
    xscale=:log10, yscale=:log10, 
    ylims=[1e-3, 1e3], legend=:topleft
    )

nlogn_refs = 0.05 * exp_timings[2,end] / (ns_both[end]*log(ns_both[end])) * ns_both .* log.(ns_both)
plot!(pl, ns_both[4:end], nlogn_refs[4:end], line=(2, :black, :dash))

n2_refs = 2.7 * exp_timings[1,4] / (ns_both[4]^2) * ns_both .^ 2
plot!(pl, ns_both[2:4], n2_refs[2:4], line=(2, :black, :dashdot))
display(pl)
savefig(pl, "./figures/both_scaling.pdf")

# plot scaling for various orders nu
colors = palette(["#FF8A8E", "#6eccc9"], length(nus))[1:end]'

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
    ns_both, roots_timings[1, 1, 2, :],  markerstrokewidth=0,
    line=(2, scrungle[1]), marker=(3, scrungle[1]), label="Direct"
    )
display(pl)
savefig(pl, "./figures/nu_scaling.pdf")

# plot scaling for various tolerances epsilon
colors = reshape(["#ffccce", palette(["#FF8A8E", "#800004"], length(tols)-1)[1:end]...], 1, :)

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
    ns_both, roots_timings[1, 1, 2, :],  markerstrokewidth=0,
    line=(2, scrungle[1]), marker=(3, scrungle[1]), label="Direct"
    )
display(pl)
savefig(pl, "./figures/tol_scaling.pdf")

# test accuracy as a function of tolerance
if isfile("./accuracy.jld")
    jld = load("./accuracy.jld")
    nu       = jld["nu"]
    n_trials = jld["n_trials"]
    tols     = jld["tols"]
    ns_acc   = jld["ns"]
    gs_true  = jld["gs_true"]
    gs_nufht = jld["gs_nufht"]
else
    nu = 0
    n_trials = 1
    tols = 10.0 .^ (-4:-1:-15)
    ns_acc = 10 .^ [3, 5, 7]
    gs_true  = [fill(NaN, n) for n in ns_acc, _=1:n_trials]
    gs_nufht = [fill(NaN, n) for _ in eachindex(tols), n in ns_acc, _=1:n_trials]
    nnz = minimum(ns_acc)
    Vs  = [fill(NaN, nnz) for _=1:n_trials]
    for t=1:n_trials
        # make coefficient vector with small number of nonzeros
        Vs[t] = randn(nnz)

        for (j, n) in enumerate(ns_acc)
            cs = zeros(n)
            I  = Random.randperm(n)[1:nnz]
            cs[I] .= Vs[t]
            rs, _, ws = test_case(:roots, [n], nothing, 1, nu=nu)

            @printf(
                    "Computing direct sum with ν = %i, n = %i (trial %i of %i)\n", 
                    nu, n, t, n_trials
                    )
            gs_true[j,t] .= dir(nu, rs[I], cs[I], ws)

            for (i, tol) in enumerate(tols)
                @printf(
                    "Computing NUFHT with ν = %i, ε = %.0e, n = %i (%i of %i, trial %i of %i)\n", 
                    nu, tol, n, i, length(tols), t, n_trials
                    )

                FastHankelTransform.setup_nufht!(nu, tol)

                gs_nufht[i,j,t] .= nufht(nu, rs, cs, ws, tol=tol)
            end
        end
    end
    save(
        "./accuracy.jld",
        "nu", nu,
        "n_trials", n_trials,
        "tols", tols,
        "ns", ns_acc,
        "gs_true", gs_true, 
        "gs_nufht", gs_nufht
        )
end

# plot accuracy as a function of tolerance
max_errs_all = [norm(gs_true[j,t] - gs_nufht[i,j,t], Inf) / norm(Vs[t], 1) for j in eachindex(ns_acc), i in eachindex(tols), t=1:n_trials]
l2_errs_all = [norm(gs_true[j,t] - gs_nufht[i,j,t]) / norm(gs_true[j]) for j in eachindex(ns_acc), i in eachindex(tols), t=1:n_trials]

l2_errs  = reshape(sum(l2_errs_all,  dims=3) / n_trials, length(ns_acc), :)'
max_errs = reshape(sum(max_errs_all, dims=3) / n_trials, length(ns_acc), :)'

pl = plot(
    tols, l2_errs,
    line=(2, scrungle'), marker=(3, scrungle'), markerstrokewidth=0,
    xlabel=L"$\varepsilon$", 
    ylabel=L"rel. $\ell^2$ error",
    label=reshape([latexstring(@sprintf("\$n=10^{%i}\$", Int(log10(n)))) for n in ns_acc], 1, :),
    xscale=:log10, yscale=:log10, 
    ylims=(5e-16, 5e-4), xlims=(5e-16, 5e-4),
    legend=:bottomright
    )
plot!([1e-16, 1e-2], [1e-16, 1e-2], line=(1, :black, :dash))
display(pl)
savefig(pl, "./figures/accuracy.pdf")