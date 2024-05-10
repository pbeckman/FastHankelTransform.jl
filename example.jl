using LinearAlgebra, SpecialFunctions, FINUFFT, QuadGK, Plots, Plots.Measures, LaTeXStrings, Printf, BenchmarkTools, FastGaussQuadrature, TimerOutputs

const TIMER = TimerOutput()

function number_boxes(n, m, boxes)
    M = fill(NaN, n, m)

    for (box_set, val) in zip(boxes, [2,3,1])
        for (b, box) in enumerate(box_set)
            i0b, i1b, j0b, j1b   = box
            M[i0b:i1b, j0b:j1b] .= val + 0.4*b/length(box_set) # min(i1b-i0b+1, j1b-j0b+1) / rank(besselj.(nu, ws[i0b:i1b]*rs[j0b:j1b]'))
        end
    end

    return M
end

##

include("./nufht.jl")

nu = 0
n  = 1_000
m  = 1_000

case = :bad

if case == :one
    m = 1
    rs = [1.0]
    ws = 10 .^ range(-6, stop=2, length=n)
elseif case == :bad
    ws = 10 .^ range(log10(0.25), stop=2, length=n)
    # rs = collect(range(0.25, stop=100, length=m))
    rs = 10 .^ range(log10(0.25), stop=2, length=m)
elseif case == :roots
    ws = FastGaussQuadrature.approx_besselroots(nu, n+1)
    rs = ws[1:end-1] / ws[end] 
    ws = ws[1:end-1]
elseif case == :twodim
    ws_1D = range(0, stop=1, length=round(Int64, sqrt(n)))
    ws    = sort([norm(v.-[0.5,0.5]) for v in vec(collect.(Iterators.product(ws_1D, ws_1D)))])
    unique!(ws)
    n      = length(ws)
    aa, bb = [0.00e+00, 1e3]
    rs, cs = gausslegendre(m)
    rs  .= (rs .+ 1) * (bb-aa)/2 .+ aa
    cs .*= (bb-aa)/2
elseif case == :twodimrandom
    ws    = sort(norm.(eachcol(rand(2,n))))
    aa, bb = [0.00e+00, 1e3]
    rs, cs = gausslegendre(m)
    rs  .= (rs .+ 1) * (bb-aa)/2 .+ aa
    cs .*= (bb-aa)/2
elseif case == :loc
    ws = collect(range(0, stop=4, length=n))
    rs = collect(range(0, stop=4, length=m))
elseif case == :lower 
    ws = collect(range(0, stop=10, length=n))
    rs = collect(range(0, stop=10, length=m))
elseif case == :vert 
    ws = collect(range(4, stop=6, length=n))
    rs = collect(range(0, stop=10, length=m))
elseif case == :horiz 
    ws = collect(range(0, stop=10, length=n))
    rs = collect(range(4, stop=6, length=m))
elseif case == :upper 
    ws = collect(range(4, stop=10, length=n))
    rs = collect(range(4, stop=10, length=m))
elseif case == :asy 
    ws = collect(range(6, stop=100, length=n))
    rs = collect(range(6, stop=100, length=m))
else
    error("case not recognized!")
end

cs = randn(m)
if max(n,m) <= 1000
    M = besselj.(nu, ws*rs')
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

# @time gs_asy  = nufht_asy(nu, rs, cs, ws, K=5)
# @time gs_tay  = nufht_tay(nu, rs, cs, ws, K=30)
# @time gs_wimp = nufht_wimp(rs, cs, ws,    K=50)
println("NUFHT:")
@time gs_nufht = nufht(nu, rs, cs, ws, min_box_dim=100, K_asy=5, K_loc=30)

pl = plot(
    xlabel=L"\omega", ylabel=L"J_\nu(\omega r)",
    ylims=1.1.*extrema(gs_nufht)
    )
scatter!(pl,
    ws, gs_nufht, 
    label="NUFHT",
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

if max(n, m) <= 10000
    gs_dir = zeros(Float64, n)
    println("Direct:")
    @time add_dir!(gs_dir, nu, rs, cs, ws)
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
    # plot!(pl,
    #     ws,
    #     abs.((gs_dir - gs_asy) ./ gs_dir) .+ 1e-16,
    #     label="Asymptotic",
    #     color=:red
    #     )
    # plot!(pl,
    #     ws,
    #     abs.((gs_dir - gs_tay) ./ gs_dir) .+ 1e-16,
    #     label="Taylor",
    #     color=:green
    #     )
    # plot!(pl,
    #     ws,
    #     abs.((gs_dir - gs_wimp) ./ gs_dir) .+ 1e-16,
    #     label="Wimp",
    #     color=:orange
    #     )
    plot!(pl,
        ws,
        abs.((gs_dir - gs_nufht) ./ gs_dir) .+ 1e-16,
        label="NUFHT",
        color=:purple
        )
    display(pl)
end
