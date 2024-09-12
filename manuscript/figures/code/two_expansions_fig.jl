using FastHankelTransform, FastGaussQuadrature, SpecialFunctions, Plots, Plots.Measures, LaTeXStrings

include("/Users/beckman/.julia/config/custom_colors.jl")

nu = 0
tol = 1e-12

FastHankelTransform.setup_nufht!(nu, tol)
# hack to evaluate NUFFTs to full precision
FastHankelTransform.NUFHT_TOL[] = 1e-15

rs = [1.0]
cs = [1.0]
n = 1000
ws = collect(range(0, 100, n))

gs_dir = besselj.(nu, ws)

gs_asy = zeros(Float64, n)
FastHankelTransform.add_asy!(gs_asy, nu, rs, cs, ws; K=FastHankelTransform.NUFHT_ASY_K[])

gs_loc = zeros(Float64, n)
FastHankelTransform.add_loc!(gs_loc, nu, rs, cs, ws; K=FastHankelTransform.NUFHT_LOC_K[])

gr(size=(400, 300))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)

pl = plot(
    ws, gs_dir, 
    line=(3, :black), xlabel=L"z", ylabel=L"J_0(z)", label="",
    ylims=[minimum(gs_dir)-0.02, 1.02]
    )
savefig("~/Downloads/bessel_function.pdf")

pl = plot(
    xlabel=L"z", ylabel="Relative error", 
    yscale=:log10, ylims=[1e-16, 1e2], legend=:right
    )
plot!(pl,
    ws, abs.((gs_dir .- gs_asy) ./ gs_dir), label="Asymptotic",
    line=(3, scrungle[1]),
    )
plot!(pl,
    ws, abs.((gs_dir .- gs_loc) ./ gs_dir), 
    line=(3, scrungle[2]), label="Local"
    )
plot!(pl,
    [FastHankelTransform.NUFHT_Z_SPLIT[], FastHankelTransform.NUFHT_Z_SPLIT[]],
    [1e-16, 1e2], line=(3, :dot, :gray25), label=""
    )
savefig("~/Downloads/pointwise_errors.pdf")
