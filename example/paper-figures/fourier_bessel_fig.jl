using FastHankelTransform, FastGaussQuadrature, FINUFFT, FFTW, SpecialFunctions, BenchmarkTools, Plots, Plots.Measures, LaTeXStrings, Printf, DelimitedFiles, Random, LinearAlgebra, JLD

# set up plotting
gr(size=(350,300))
default(margins=1mm, fontfamily="Computer Modern", label="")
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)

function quad1D(m, a, b)
    # Gauss-Legendre on [a,b]
    rs, wts = gausslegendre(m)
    rs   .= (rs .+ 1) * (b-a)/2 .+ a
    wts .*= (b-a)/2

    return rs, wts
end

# we need a precomputed table with 10,000 roots of orders nu=0,...,1000.
# i'm not aware of a Julia package that does this accurately...
# these roots were computed using code from Michael O'Neil implementing:
#
# Glaser, Andreas, Xiangtao Liu, and Vladimir Rokhlin. 
# "A fast algorithm for the calculation of the roots of special functions."
# SIAM Journal on Scientific Computing 29, no. 4 (2007): 1420-1438.
bessel_roots = readdlm("./bessel_roots.txt")

function compute_FB_coefficients(
    k, cs, rs, tol; wts=nothing, adjoint=false
    )
    if !adjoint && isnothing(wts)
        error("to compute the forward transform, you must provide quadrature weights.")
    end
    mr = length(rs)
    ws = bessel_roots[abs(k)+1, 1:mr]
    bi = Vector{ComplexF64}(undef, mr)
    if !adjoint
        # compute integral using forward transform with weights
        bi  .=    nufht(k, rs, real.(cs) .* rs .* wts, ws, tol=tol)
        bi .+= im*nufht(k, rs, imag.(cs) .* rs .* wts, ws, tol=tol)
        bi .*= 2 ./ besselj.(abs(k)+1, ws).^2
    else
        # compute summation using adjoint transform without weights
        bi  .=    nufht(k, ws, real.(cs), rs, tol=tol)
        bi .+= im*nufht(k, ws, imag.(cs), rs, tol=tol)
    end

    return bi
end

function compute_FB_series(f; tol=1e-8)
    converged = false
    mt, mr = 2, 9
    rs, wts = quad1D(mr, 0, 1)
    ts, ks = [], []
    as, last_as = [], []
    # refine in theta until Fourier coefficients converge
    while !converged
        if mt > 200
            error("trapezoidal rule suggests more than 200 Fourier-Bessel modes are needed... NUFHT not implemented for these high orders")
        end
        mt *= 2
        ts = range(0, 2pi, 2mt+2)[1:end-1]

        ks = -mt:mt
        as = 1/2pi * 2pi/(2mt+1) * fftshift(fft(f(rs, ts'), 2), 2)

        if !isempty(last_as)
            # column subindices from last iteration
            inds  = (mt+1-div(mt, 2)):(mt+1+div(mt, 2))
            # compute norm difference with coefficients at last iteration 
            # as a proxy for discretization error
            err   = norm(as[:,inds] .- last_as) / norm(as)
            # compute norm of coefficients at this iteration
            # as a proxy for truncation error
            trunc = norm(as[:,1:mt-div(mt,2)]) .+ norm(as[:,mt+2+div(mt,2):end]) / norm(as)
            converged = all([err, trunc] .< tol)
            @printf("Using %i trapezoidal nodes gives discretization error %.2e and truncation error %.2e\n", mt, err, trunc)
        end

        last_as = as

        # roll back last doubling since its contribution was less than tol
        if converged
            inds = (mt+1-div(mt, 2)):(mt+1+div(mt, 2))
            as   = as[:,inds]

            mt = div(mt, 2)
            ts = range(0, 2pi, 2mt+2)[1:end-1]
            ks = -mt:mt
        end
    end
    println()

    converged = false
    bs, last_bs = [], []
    # refine in r
    while !converged
        mr = 2*(mr-1) + 1
        rs, wts = quad1D(mr, 0, 1)

        as = 1/2pi * 2pi/(2mt+1) * fftshift(fft(f(rs, ts'), 2), 2)

        bs = Matrix{ComplexF64}(undef, mr, 2mt+1)
        for (i, k) in enumerate(ks)
            bs[:,i] .= compute_FB_coefficients(
                k, as[:,i], rs, tol/100, wts=wts, adjoint=false
                )
        end

        if !isempty(last_bs)
            # self-convergence error in computed coefficients
            err   = norm(bs[1:div(mr-1,2)+1,:] .- last_bs) / norm(bs)
            # self-convergence truncation error
            trunc = norm(bs[div(mr-1,2)+2:end,:]) / norm(bs)
            converged = all([err, trunc] .< tol)
            @printf("Using %i Gauss-Legendre nodes gives discretization error %.2e and truncation error %.2e\n", mr, err, trunc)
        end

        last_bs = bs

        # roll back last doubling since its contribution was less than tol
        if converged
            bs = bs[1:div(mr-1,2)+1,:]
            mr = div(mr-1, 2) + 1
            rs, _ = quad1D(mr, 0, 1)
        end
    end
    
    return bs, rs, ts
end

function dense_eval_FB_series(bs, rs, ts)
    mr = size(bs, 1)
    mt = div(size(bs, 2)-1, 2)
    ks = -mt:mt

    fs = zeros(ComplexF64, length(rs), length(ts))
    for (i, k) in enumerate(ks)
        for j=1:mr
            fs .+= bs[j,i] * besselj.(k, bessel_roots[abs(k)+1,j]*rs) * exp.(im*k*ts)
        end
    end

    return fs
end

function eval_FB_series(bs; tol=1e-8)
    mr = size(bs, 1)
    mt = div(size(bs, 2)-1, 2)
    rs, _ = quad1D(mr, 0, 1)
    ks = -mt:mt

    ts = range(0, 2pi, 2mt+2)[1:end-1]

    gs = Matrix{ComplexF64}(undef, size(bs))
    for (i, k) in enumerate(ks)
        gs[:,i] .= compute_FB_coefficients(k, bs[:,i], rs, tol, adjoint=true)
    end

    fs = ifft(ifftshift(gs, 2), 2) * (2mt+1)
    
    return fs, rs, ts
end

# define random forcing f
taper(r, r0) = r > 1 ? NaN : (1 - r^2)*(1 + 16r^2/43 - r^4/43) / (1 + 4r^2)
Random.seed!(0)
nbl = 100
tbl = 2pi * rand(nbl)
cs = log.(1 .+ 2rand(nbl)) .* [cos.(tbl) sin.(tbl)]
vs = randn(nbl)
ps = 0.15 .+ log.(1 .+ 0.2rand(nbl))
f(r::Float64,t::Float64) = 100 * taper(r, 0) * sum(
    v * exp(-(norm([r*cos(t), r*sin(t)] - c) / p)^2) for (v, c, p) in zip(vs, eachrow(cs), ps)
)
f(rs::AbstractArray{Float64}, ts::AbstractArray{Float64}) = f.(rs, ts)

# compute Fourier-Bessel coefficients of f to given tolerance
tol = 1e-6
bs, rs, ts = compute_FB_series(f, tol=tol)

# evaluate Fourier-Bessel approximation to f at tensor product points
fs_FB_mat, rs_pl, ts_pl = eval_FB_series(bs)

# evaluate true f at the same points
fs_mat = f(rs_pl, ts_pl')

@printf("relative error in Fourier-Bessel approximation of f = %.2e\n", norm(fs_mat .- fs_FB_mat) / norm(fs_mat))

# plot forcing f
clim_f = maximum(abs.(fs_mat))
pl = heatmap(
    ts, rs, real.(fs_mat), dpi=300,
    c=reverse(cgrad(:RdYlBu)), yflip=true, 
    projection=:polar, grid=false, axis=false,
    clims=(-clim_f, clim_f), rightmargin=4mm
    )
display(pl)
savefig(pl, "./figures/helmholtz_f.pdf")

# plot Fourier-Bessel reconstruction of f
clim_f = maximum(abs.(fs_FB_mat))
pl = heatmap(
    ts, rs, real.(fs_FB_mat), dpi=300,
    c=reverse(cgrad(:RdYlBu)), yflip=true, 
    projection=:polar, grid=false, axis=false,
    clims=(-clim_f, clim_f), rightmargin=4mm
    )
display(pl)
savefig(pl, "./figures/helmholtz_f_FB.pdf")

# plot Fourier-Bessel coefficient magnitudes
mr = size(bs, 1)
mt = div(size(bs, 2)-1, 2)
ks = -mt:mt
pl = heatmap(
    ks, 1:mr, log10.(abs.(bs)), clims=(-8, 0), dpi=300,
    xlabel=L"\ell", ylabel=L"j",
    yflip=true, c=palette(:Purples),
    margins=5mm
    )
display(pl)
savefig(pl, "./figures/helmholtz_f_coef.pdf")

# compute eigenvalues λ_m of Dirichlet Laplacian on the disk
lm = Matrix{Float64}(undef, size(bs))
for (i, k) in enumerate(ks)
    for j=1:mr
        lm[j,i] = -bessel_roots[abs(k)+1,j]^2
    end
end

# set wavenumber κ and evaluate Helmholtz solution
kp = 25
us_FB_mat, rs, ts = eval_FB_series(-bs ./ (kp^2 .+ lm))

# plot solution
clim_u = maximum(abs.(us_FB_mat))
pl = heatmap(
    ts, rs, real.(us_FB_mat), dpi=300,
    c=reverse(cgrad(:RdYlBu)), yflip=true, 
    projection=:polar, grid=false, axis=false,
    clims=(-clim_u, clim_u),
    rightmargin=4mm
    )
display(pl)
savefig(pl, "./figures/helmholtz_u_kappa$kp.pdf")
