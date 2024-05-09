using LinearAlgebra, SpecialFunctions, FINUFFT, QuadGK, Plots, Plots.Measures, LaTeXStrings, Printf, BenchmarkTools, FastGaussQuadrature

include("./nufht.jl")

nu = 0
n  = 1000
m  = 1000

case = :bad

if case == :one
    m = 1
    rs = [1.0]
    ws = 10 .^ range(-6, stop=2, length=n)
elseif case == :log
    ws = 10 .^ range(-6, stop=0, length=n)
    rs = sort(1000rand(m))
elseif case == :bad
    ws = 10 .^ range(log10(0.25), stop=2, length=n)
    # rs = collect(range(0.25, stop=100, length=m))
    rs = 10 .^ range(log10(0.25), stop=2, length=m)
elseif case == :roots
    ws = FastGaussQuadrature.approx_besselroots(nu, n+1)
    rs = ws[1:end-1] / ws[end] 
    ws = ws[1:end-1]
elseif case == :lows
    ws = sort(2rand(n))
    rs = sort(2rand(m))
elseif case == :mids
    ws = sort(1 .+ 0.1rand(n))
    rs = sort(10 .+ 100rand(m))
elseif case == :highs
    ws = sort(rand(n))
    rs = sort(rand(m))
elseif case == :twodim
    ws_1D = range(0, stop=1, length=round(Int64, sqrt(n)))
    ws    = sort(vec(norm.(collect.(Iterators.product(ws_1D, ws_1D)))))
    n     = length(ws_1D)^2
    aa, bb = [0.00e+00, 1e2]
    rs, cs = gausslegendre(m)
    rs  .= (rs .+ 1) * (bb-aa)/2 .+ aa
    cs .*= (bb-aa)/2
elseif case == :twodimrandom
    ws    = sort(norm.(eachcol(rand(2,n))))
    aa, bb = [0.00e+00, 1e4]
    rs, cs = gausslegendre(m)
    rs  .= (rs .+ 1) * (bb-aa)/2 .+ aa
    cs .*= (bb-aa)/2
else
    error("case not recognized!")
end
cs = randn(m)
M = besselj.(nu, ws*rs')
if max(n,m) < 1000
    @show rank(M)
end

# ws = 10 .^ range(-4, stop=0, length=n)

# aa, bb = [0.00e+00, 1e4]
# aa, bb = [1e3, 1e4]

# aa, bb = [0.00e+00, 2.05e+05]

# aa, bb = [2.05e+05, 4.10e+05]

# aa, bb = [1.06e+06, 1.51e+06]
# ws = ws[ws .< 4.57e-03]

# aa, bb = [3.22e+07, 1.84e+08]
# ws = ws[ws .< 1.35e-05]

# aa, bb = [2.36e+09, 1.15e+11]
# ws = ws[ws .< 1.81e-08]

# rs, cs = gausslegendre(m)
# rs  .= (rs .+ 1) * (bb-aa)/2 .+ aa
# cs .*= (bb-aa)/2

@time gs_asy  = nufht_asy(nu, rs, cs, ws, K=5)
@time gs_tay  = nufht_tay(nu, rs, cs, ws, K=30)
@time gs_wimp = nufht_wimp(rs, cs, ws,    K=50)
@time gs_nufht = nufht(rs, cs, ws, min_box_dim=200)

if max(n, m) < 10000
    @time gs_dir  = nufht_dir(nu, rs, cs, ws)
    pl = plot(
        xlabel=L"\omega", ylabel=L"J_\nu(\omega r)",
        ylims=1.1.*extrema(gs_dir)
        )
    scatter!(pl,
        ws, gs_dir, 
        label="Direct",
        # markerstrokecolor=:grey, 
        markercolor=:black, 
        markersize=2
        )
    # scatter!(pl,
    #     ws, gs_res,
    #     label="Residue",
    #     markerstrokecolor=:blue, 
    #     markercolor=:blue, 
    #     markersize=4
    #     )
    # scatter!(pl,
    #     ws, gs_asy,
    #     label="Asymptotic",
    #     markerstrokecolor=:red, 
    #     markercolor=:red, 
    #     markersize=2
    #     )
    display(pl)

    pl = plot(
        xlabel=L"\omega",
        ylabel="r",
        yscale=:log10,
        ylims=[1e-17, 1e1],
        legend=:topright
    )
    # plot!(pl,
    #     ws,
    #     abs.((gs_dir - gs_res) ./ gs_dir) .+ 1e-16,
    #     label="Residue",
    #     color=:blue
    #     )
    plot!(pl,
        ws,
        abs.((gs_dir - gs_asy) ./ gs_dir) .+ 1e-16,
        label="Asymptotic",
        color=:red
        )
    plot!(pl,
        ws,
        abs.((gs_dir - gs_tay) ./ gs_dir) .+ 1e-16,
        label="Taylor",
        color=:green
        )
    plot!(pl,
        ws,
        abs.((gs_dir - gs_wimp) ./ gs_dir) .+ 1e-16,
        label="Wimp",
        color=:orange
        )
    plot!(pl,
        ws,
        abs.((gs_dir - gs_nufht) ./ gs_dir) .+ 1e-16,
        label="Asympt + Direct",
        color=:purple
        )
    display(pl)
end
