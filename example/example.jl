
using Random, LinearAlgebra, SpecialFunctions, FastHankelTransform, Plots, Plots.Measures, LaTeXStrings, FastGaussQuadrature, Printf, BenchmarkTools, TimerOutputs
# using FINUFFT

include("./test_cases.jl")

# order of transform
nu  = 4 + 1/2
# number of sources
m   = 1_000
# number of targets
n   = 1_000
# tolerance
tol = 1e-8

FastHankelTransform.setup_nufht!(nu, tol)

K_asy = FastHankelTransform.NUFHT_ASY_K[]
K_loc = FastHankelTransform.NUFHT_LOC_K[]

# case = :twodimrandom
# case = :roots
case = :one

Random.seed!(123)
rs, cs, ws, n, m = test_case(case, n, m)

println("Benchmarking NUFHT...")
# @btime gs_nufht = nufht($nu, $rs, $cs, $ws, tol=$tol, max_levels=5)
@time gs_nufht = nufht(nu, rs, cs, ws, tol=tol, min_dim_prod=n-1);

if isinteger(nu)
    dims(box) = (box[2] - box[1] + 1, box[4] - box[3] + 1)
    boxes = FastHankelTransform.generate_boxes(
        rs, ws, z_split=FastHankelTransform.NUFHT_Z_SPLIT[]
        )
    ratios = [sum(prod.(dims.(box_set))) for box_set in boxes] / (n*m)
    @printf("Portion of entries by expansion:  
    Local:      %.6f
    Asymptotic: %.6f
    Direct:     %.6f\n", ratios...)
end

# println("Benchmarking 1D NUFFT...")
# ccs = ComplexF64.(cs)
# # @btime gs_nufft = nufft1d3($rs, $ccs, +1, $tol, $ws)
# @time gs_nufft = nufft1d3(rs, ccs, +1, tol, ws)

if max(m, n) <= 1000
    gs_dir = zeros(Float64, n)
    println("Direct:")
    @time FastHankelTransform.add_dir!(gs_dir, nu, rs, cs, ws)

    @printf("\nRelative error : %.2e\n\n", norm(gs_dir - gs_nufht) / norm(gs_dir))
end

##

if max(n,m) <= 1000
    M = besselj.(nu, ws*rs')
    @show rank(M)
end

# println("NUFHT:")
# @time gs_nufht = nufht(nu, rfs, cs, ws, tol=tol)

gr(size=(1000,700))
default(margins=2mm, fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)

if max(n, m) <= 10000
    if case == :one
        pl = plot(
            title="ν = $nu",
            xlabel=L"z",
            ylabel=L"relative error in $J_\nu(z)$",
            yscale=:log10,
            ylims=[1e-17, 1e1],
            # legend=:top
        )
        # println("Asymptotic:")
        # gs_asy = zeros(Float64, n)
        # @time FastHankelTransform.add_asy!(gs_asy, nu, rs, cs, ws, K=K_asy)
        # println("Wimp:")
        # gs_wimp = zeros(Float64, n)
        # @time FastHankelTransform.add_loc!(gs_wimp, nu, rs, cs, ws, K=K_loc)
        # println("Taylor:")
        # gs_tay = zeros(Float64, n)
        # @time add_taylor!(gs_tay, nu, rs, cs, ws, K=10)
        # plot!(pl,
        #     ws,
        #     abs.((gs_dir - gs_wimp) ./ gs_dir) .+ 1e-16,
        #     label="Wimp",
        #     # color=cgrad(:default, 5, categorical=true)[2],
        #     line=3
        #     )
        # plot!(pl,
        #     ws,
        #     abs.((gs_dir - gs_asy) ./ gs_dir) .+ 1e-16,
        #     label="Hankel",
        #     # color=cgrad(:default, 5, categorical=true)[4],
        #     line=3
        #     )
        # plot!(pl,
        #     ws,
        #     abs.((gs_dir - gs_tay) ./ gs_dir) .+ 1e-16,
        #     label="Taylor",
        #     # color=cgrad(:default, 5, categorical=true)[3],
        #     line=2
        #     )
        # if nu > 0
        #     println("Debye:")
        #     gs_deb = zeros(Float64, n)
        #     @time FastHankelTransform.add_debye!(gs_deb, nu, rs, cs, ws, K=3, large_z=false)
        #     println("Debye z/ν → ∞:")
        #     gs_debz = zeros(Float64, n)
        #     @time FastHankelTransform.add_debye!(gs_debz, nu, rs, cs, ws, K=3, large_z=true, Kz=16)
        #     # gs_deb[ws .>= nu] .= NaN
        #     plot!(pl,
        #         ws,
        #         abs.((gs_dir - gs_deb) ./ gs_dir) .+ 1e-16,
        #         label="Debye",
        #         # color=cgrad(:default, 5, categorical=true)[3],
        #         line=2
        #         )
        #     plot!(pl,
        #         ws,
        #         abs.((gs_dir - gs_debz) ./ gs_dir) .+ 1e-16,
        #         label=L"Debye $(\frac{z}{\nu} \to \infty)$:",
        #         # color=cgrad(:default, 5, categorical=true)[3],
        #         line=2
        #         )
        # end
        plot!(pl,
            ws,
            abs.((gs_dir - gs_nufht) ./ gs_dir) .+ 1e-16,
            label="NUFHT",
            color=:purple,
            line=(3, :dash)
        )
        plot!(pl,
            ws,
            abs.((gs_dir - (ws/2).^nu/gamma(nu + 1)) ./ gs_dir) .+ 1e-16,
            label="power law",
            color=:red,
            line=(3, :solid)
        )
        display(pl)
    end
end

##

gr(size=(500,500))
z = 25
pl = heatmap(
    number_boxes(
        n, m, generate_boxes(rs, ws, z, min_box_dim=2, max_levels=4)
        ), 
    yflip=true, 
    xticks=((),()), yticks=((),()), legend=:none,
    ylabel=L"r", xlabel=L"\omega"
)