using FastHankelTransform, FastGaussQuadrature, BenchmarkTools, Plots, Printf

include("./test_cases.jl")

ns  = 10 .^ range(3, 7, 10)
ns_both = 2 .^ (8:2:24)
ps  = 10 .^ range(5, 9, 10)
nu  = 0
tol = 1e-8

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

## Test scaling with n, m, and P

n_timings = fill(NaN, length(ns))
m_timings = fill(NaN, length(ns))
p_timings = fill(NaN, length(ps))
for (case, timings) in zip([:n, :m, :p], [n_timings, m_timings, p_timings])
    for i in eachindex(timings)
        rs, cs, ws, n, m, p = test_case(case, ns, ps, i)

        @printf(
            "Timing %s scaling with n = %i, m = %i, p = %.1e\n", 
            string(case), n, m, p
            )

        timings[i] = @belapsed nufht($nu, $rs, $cs, $ws, tol=tol)
    end
end

## Test scaling for scaled roots and exponential

roots_timings = fill(NaN, length(ns_both))
exp_timings   = fill(NaN, length(ns_both))
for (case, timings) in zip([:roots, :exp], [roots_timings, exp_timings])
    for i in eachindex(ns_both)
        rs, cs, ws, n, m, p = test_case(case, ns_both, nothing, i)

        @printf(
            "Timing %s scaling with n = %i, p = %.1e\n", 
            string(case), n, p
            )

        timings[i] = @belapsed nufht($nu, $rs, $cs, $ws, tol=tol)
    end
end

##

using DelimitedFiles

timings = readdlm("nufht_timing.csv")

gr(size=(500,500))
default(margins=2mm, fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)

pl = plot(
    xlabel="n", ylabel="time (s)", 
    xscale=:log10, yscale=:log10, 
    ylims=[0.5minimum(timings[(!).(isnan.(timings))]), 1e2],
    legend=:bottomright
    )
plot!(pl, 
        ns, timings', 
        label=[reshape(string.(cases), 1, 4) "direct"], marker=4, line=1,
        markershape=[:circle :rect :diamond :cross :utriangle],
        markerstrokewidth=0
    )
display(pl)

i = div(length(ns), 2)
c = timings[1,i] / (ns[i]*log2(ns[i]))
plot!(
    ns, 1.5 * c * ns .* log2.(ns), 
    label="", line=(2, :dash, :black)
    )
annotate!(
    8e4, 6e1, text(L"\mathcal{O}(n\log n)", 14)
)

c = timings[3,i] / ns[i]
plot!(
    ns, 1.5 * c * ns, 
    label="", line=(2, :dash, :black)
    )
annotate!(
    1e4, 3e-1, text(L"\mathcal{O}(n)", 14)
)

id = div(length(ns) - sum(isnan.(timings[end,:])), 2)
cd = timings[end,id] / ns[id]^2
plot!(
    ns, 1.5 * cd * ns.^2, 
    label="", line=(2, :dash, :black)
    )
annotate!(
    3.5e3, 1e1, text(L"\mathcal{O}(n^2)", 14)
)