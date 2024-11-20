using FastHankelTransform, FastGaussQuadrature, FINUFFT, FFTW, SpecialFunctions, BenchmarkTools, Plots, Plots.Measures, LaTeXStrings, Printf, DelimitedFiles, Random, LinearAlgebra, JLD

include("/Users/beckman/.julia/config/custom_colors.jl")

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

bessel_roots = readdlm("/Users/beckman/Downloads/bessel_roots.txt")

##

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
        @show size(ws), size(cs), size(rs)
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
        @show mt
        ts = range(0, 2pi, 2mt+2)[1:end-1]

        ks = -mt:mt
        as = 1/2pi * 2pi/(2mt+1) * fftshift(fft(f(rs, ts'), 2), 2)

        if !isempty(last_as)
            # column subindices from last iteration
            inds  = (mt+1-div(mt, 2)):(mt+1+div(mt, 2))
            # compute norm difference with coefficients at last iteration 
            # as a proxy for discretization error
            err   = norm(as[:,inds] .- last_as) / norm(as)
            # compute norm of coefficients add this iteration
            # as a proxy for truncation error
            trunc = norm(as[:,1:mt-div(mt,2)]) .+ norm(as[:,mt+2+div(mt,2):end]) / norm(as)
            @show err, trunc
            converged = all([err, trunc] .< tol)
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

    converged = false
    bs, last_bs = [], []
    # refine in r
    while !converged
        mr = 2*(mr-1) + 1
        @show mr
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
            @show err, trunc
            converged = all([err, trunc] .< tol)
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

##

example = :tapered
# example = :bandlimited

if example == :tapered
    # taper(r, r0) = 1

    # taper(r, r0) = r > 1 ? NaN : (1 - r^2) / (4 + r^2)

    # taper(r, r0) = r > 1 ? NaN : (1 - r^2) / (1 + r^2)

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
elseif example == :bandlimited
    max_j = 100
    max_l = 30
    p = ([norm([j/max_j, l/max_l]) for j=0:max_j, l=-max_l:max_l] / sqrt(2)) .+ 0.3
    bs_true = (rand(max_j+1, 2max_l+1) .> p) .* (randn(max_j+1, 2max_l+1) + im*randn(max_j+1, 2max_l+1)) / sqrt(2)
    
    f(rs::AbstractArray, ts::AbstractArray) = dense_eval_FB_series(
        bs_true, rs, ts
        )
else
    error("example not found.")
end

rs = range(0, 1, 200)
ts = range(0, 2pi, 200)[1:end-1]

fs_mat = f(rs, ts')

clim_f = maximum(abs.(fs_mat))
pl = heatmap(
    ts, rs, real.(fs_mat),
    c=reverse(cgrad(:RdYlBu)), yflip=true, 
    projection=:polar, grid=false, axis=false,
    clims=(-clim_f, clim_f), rightmargin=4mm
    )
display(pl)

##

tol = 1e-6
bs, rs, ts = compute_FB_series(f, tol=tol)

##

mr = size(bs, 1)
mt = div(size(bs, 2)-1, 2)
ks = -mt:mt

bs_pl = zeros(ComplexF64, mr, 1025)
bs_pl[:, (513-mt):(513+mt)] .= bs

fs_FB_mat, rs_pl, ts_pl = eval_FB_series(bs_pl)

fs_mat  = f(rs_pl, ts_pl')

@show norm(fs_mat .- fs_FB_mat) / norm(fs_mat)

##

clim_f = maximum(abs.(fs_mat))
pl = heatmap(
    ts, rs, real.(fs_mat), dpi=300,
    c=reverse(cgrad(:RdYlBu)), yflip=true, 
    projection=:polar, grid=false, axis=false,
    clims=(-clim_f, clim_f), rightmargin=4mm
    )
display(pl)
savefig(pl, "~/Downloads/helmholtz_f.pdf")

clim_f = maximum(abs.(fs_FB_mat))
pl = heatmap(
    ts, rs, real.(fs_FB_mat), dpi=300,
    c=reverse(cgrad(:RdYlBu)), yflip=true, 
    projection=:polar, grid=false, axis=false,
    clims=(-clim_f, clim_f), rightmargin=4mm
    )
display(pl)
savefig(pl, "~/Downloads/helmholtz_f_FB.pdf")

pl = heatmap(
    ks, 1:mr, log10.(abs.(bs)), clims=(-8, 0), dpi=300,
    # imag.(bs),
    # title=L"\log_{10}|\alpha_{jl}|", 
    xlabel=L"\ell", ylabel=L"j",
    yflip=true, c=palette(:Purples),
    margins=5mm
    )
display(pl)
savefig(pl, "~/Downloads/helmholtz_f_coef.pdf")

##

lm = Matrix{Float64}(undef, size(bs))
for (i, k) in enumerate(ks)
    for j=1:mr
        lm[j,i] = -bessel_roots[abs(k)+1,j]^2
    end
end

##

kp = 25
us_FB_mat, rs, ts = eval_FB_series(-bs ./ (kp^2 .+ lm))

##

clim_u = maximum(abs.(us_FB_mat))
pl = heatmap(
    ts, rs, real.(us_FB_mat), dpi=300,
    c=reverse(cgrad(:RdYlBu)), yflip=true, 
    projection=:polar, grid=false, axis=false,
    clims=(-clim_u, clim_u),
    rightmargin=4mm
    )
display(pl)
savefig(pl, "~/Downloads/helmholtz_u_kappa$kp.pdf")

##

err  = f_mat .- f_FB_mat
err_clim = maximum(abs.(err[(!).(isnan.(err))]))
pl = heatmap(
    log10.(abs.(err)),
    c=:lightrainbow,
    yflip=true,
    margins=5mm
    )
# pl = heatmap(
#     log10.(abs.(err)),
#     c=:lightrainbow,
#     clims=(-8, 0), 
#     margins=5mm
#     )
display(pl)