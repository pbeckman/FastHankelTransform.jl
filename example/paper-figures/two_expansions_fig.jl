using FastHankelTransform, FastGaussQuadrature, SpecialFunctions, Plots, Plots.Measures, LaTeXStrings

# order and tolerance for testing
nu  = 0
tol = 1e-12

FastHankelTransform.setup_nufht!(nu, tol)
# evaluate NUFFTs to full precision to focus on accuracy of expansion
FastHankelTransform.NUFHT_TOL[] = 1e-15

# set up to evaluate a single Bessel function at many points
rs = [1.0]
cs = [1.0]
n = 1000
ws = collect(range(0, 100, n))

# compute Bessel function directly
gs_dir = besselj.(nu, ws)

# compute using asymptotic expansion everywhere
gs_asy = zeros(Float64, n)
FastHankelTransform.add_asy!(gs_asy, nu, rs, cs, ws; K=FastHankelTransform.NUFHT_ASY_K[])

# compute using local expansion everywhere
gs_loc = zeros(Float64, n)
FastHankelTransform.add_loc!(gs_loc, nu, rs, cs, ws; K=FastHankelTransform.NUFHT_LOC_K[])

# set up plotting
gr(size=(600, 300))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)

pl = plot(
    ws, gs_dir, 
    line=(3, :black), xlabel=L"z", ylabel=L"J_0(z)", label="",
    ylims=[minimum(gs_dir)-0.02, 1.02]
    )
savefig("./figures/bessel_function.pdf")

# plot errors
pl = plot(
    xlabel=L"z", ylabel="Relative error", 
    yscale=:log10, ylims=[1e-16, 1e2], legend=:right
    )
plot!(pl,
    ws, abs.((gs_dir .- gs_asy) ./ gs_dir), label="Asymptotic",
    line=(3, scrungle[6]),
    )
plot!(pl,
    ws, abs.((gs_dir .- gs_loc) ./ gs_dir), 
    line=(3, scrungle[7]), label="Local"
    )
# plot crossover
plot!(pl,
    [FastHankelTransform.NUFHT_Z_SPLIT[], FastHankelTransform.NUFHT_Z_SPLIT[]],
    [1e-16, 1e2], line=(3, :dot, :gray25), label=""
    )
savefig("./figures/pointwise_errors.pdf")
