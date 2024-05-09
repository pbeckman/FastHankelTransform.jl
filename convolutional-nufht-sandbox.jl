using LinearAlgebra, SpecialFunctions, FINUFFT, QuadGK, Plots, Plots.Measures, LaTeXStrings, Printf

gr(size=(800, 250))
default(fontfamily="Computer Modern", margin=5mm)
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)

# set up
nu   = 0
tau  = 0.1
rk   = [2.5, 1.2, 0.15]
c    = 1
tol  = 1e-7
maxw = 100

# define H, F, regularized F, and this real space convolution
F(t)   = c/sqrt(2pi*tau^2) * exp(-t) * sum(exp.(-(t .+ log.(rk)).^2 / (2tau^2)))
H(t)   = exp(t) * besselj(nu, exp(t))
FcH(t) = quadgk(u -> F(u)*H(t-u), -log(rk)-10tau, -log(rk)+10tau)[1]

g(t)     = exp(-t^2/(2*tau^2))
g_hat(s) = sqrt(2pi)*tau * exp(-2pi^2 * s^2 * tau^2)

ts = log.(range(1e-3, stop=maxw, length=1000))
@printf("Max ω : %.2e\n", exp(ts[end]))

# define Fourier transforms of H, F, and regularized F
F_hat_true(s) = c*sum(rk .* exp.(2pi*log.(rk)*im*s))
F_hat(s; tau=tau) = c/sqrt(2pi*tau^2)*sqrt(2pi)*tau*sum(rk .* exp.(-2*pi^2*tau^2*s^2 .+ 2pi*(tau^2 .+ log.(rk))*im*s .+ tau^2/2))
H_hat(s) = 2^(-2pi*im*s)*exp(loggamma(1/2 - im*pi*s)-loggamma(1/2 + im*pi*s))
g_cH_hat(t) = quadgk(u -> g(u)*H_hat(t-u), -10tau, 10tau)[1]

# determine domain of integration in spectral space
S = 100
# while abs(F_hat(S)) > tol/1e3
#     S *= 2
# end
ss = range(-S/5, stop=S/5, length=10000)
conv_evals = g_cH_hat.(ss)
p2 = plot(
    ss,
    # [real.(H_hat.(ss)) imag.(H_hat.(ss))], 
    [real.(conv_evals) imag.(conv_evals)], 
    # [real.(F_hat.(ss) .* H_hat.(ss)) imag.(F_hat.(ss) .* H_hat.(ss))], 
    # title=@sprintf("τ = %.0e", tau),
    # labels=[L"$Re(\widehat{F}_\tau\cdot\widehat{H}_\nu)$"
    # L"$Im(\widehat{F}_\tau\cdot\widehat{H}_\nu)$"], 
    labels=[L"$Re(\widehat{G}_\tau * \widehat{H}_\nu)$" L"$Im(\widehat{G}_\tau * \widehat{H}_\nu)$"], 
    xlabel="s",
    color=[:red :blue], line=1
    )
display(p2)

##

# compute convolution adaptively in spectral space
m = 1000
FcH_fourier = zeros(ComplexF64, length(ts))
err         = zeros(ComplexF64, length(ts))
firstiter   = true
while firstiter || norm(err) / norm(FcH_fourier) > tol
    firstiter = false
    m    = round(Int64, 2m)
    h    = 2S / (m-1)
    nds  = collect((-S):h:S)
    err .= h*nufft1d3(
        nds, 
        F_hat.(nds) .* H_hat.(nds), 
        +1, 1e-15, 
        2pi*collect(ts)
        ) .- FcH_fourier
    FcH_fourier .+= err
    @printf("m = %i, norm err = %.2e\n", m, norm(err) / norm(FcH_fourier))
end
FcH_fourier ./= g.(exp.(ts))

# # compute convolution adaptively in real space
# FcH_quadgk = getindex.(FcH.(ts),1)

# compute convolution using residue theorem
K = 30
poleH_hat(k) = -((nu+1)/2 + k) * im/pi
resH_hat(k)  = (-1)^k * im/pi * 2.0^(-(2k + nu + 1))/(gamma(k+1)*gamma(nu+k+1))

FcH_res = zeros(ComplexF64, length(ts)) 
for k=0:K
    xik       = poleH_hat(k)
    term      = resH_hat(k) * F_hat_true(xik) * exp.(2pi*im*xik*ts[end])
    # @printf(
    #     "Largest contribution in term %i : %.2e\n", 
    #     k, imag(term)
    #     )
    FcH_res .+= resH_hat(k) * F_hat_true(xik) * exp.(2pi*im*xik*ts)
end
FcH_res .*= -2pi*im # why is this negative?

# @printf(
#     "relative 2-norm error in F*H : %.2e\n", 
#     norm(FcH_fourier - FcH_quadgk) / norm(FcH_quadgk)
#     )

tspl = range(-3, stop=5, length=20000)
p1 = plot(
    tspl, 
    [F.(tspl) H.(tspl)], 
    labels=[L"F_\tau" L"H_\nu"], 
    xlabel=L"t", line=2,
    legend=:topleft
    )
# plot!(p1,
#     ts,
#     real.(FcH_quadgk), 
#     label=L"(F*H)",
#     line=2
# )
# plot!(p1,
#     ts,
#     real.(FcH_fourier), 
#     label=L"\mathcal{F}^{-1}(\widehat{F}\widehat{H})",
#     line=(2, :dash)
# )
display(p1)

gr(size=(800, 500))

bessel_evals  = sum(besselj.(nu, exp.(ts)*rk') .* (rk').^2, dims=2) # why is this rk squared?
fourier_evals = exp.(-ts) .* real.(FcH_fourier) # * sqrt(2pi)*tau
residue_evals = exp.(-ts) .* real.(FcH_res)

@printf(
    "relative 2-norm error in Fourier convolution : %.2e\n", 
    norm(bessel_evals - fourier_evals) / norm(bessel_evals)
    )

@printf(
    "relative 2-norm error in residue theorem : %.2e\n", 
    norm(bessel_evals - residue_evals) / norm(bessel_evals)
    )

p3 = plot(
    title=@sprintf("τ = %.0e", tau),
    xlabel=L"\omega",
    legend=:topright,
    ylims=1.1 .* extrema(bessel_evals)
    )
plot!(p3,
    exp.(ts),
    bessel_evals,
    label="Direct evaluation",
    line=(2, :black),
)
plot!(p3,
    exp.(ts),
    fourier_evals,
    label="Fourier convolution",
    line=(2, :mediumpurple, :dash)
)
plot!(p3,
    exp.(ts),
    residue_evals,
    label="Residue theorem",
    line=(2, :orange, :dashdot)
)

fourier_errs = abs.((fourier_evals .- bessel_evals) ./ bessel_evals)
residue_errs = abs.((residue_evals .- bessel_evals) ./ bessel_evals)
p4 = plot(
    title="Relative errors",
    xlabel=L"\omega",
    legend=:bottomright,
    yscale=:log10,
    ylims=[1e-16, 1e0]
)
plot!(p4,
    exp.(ts),
    fourier_errs,
    label="Fourier convolution",
    line=(2, :mediumpurple, :dash)
)
plot!(p4,
    exp.(ts),
    residue_errs,
    label="Residue theorem",
    line=(2, :orange, :dashdot)
)
pl = plot(p3, p4, layout=grid(2,1))


#### DANGER #### TURN BACK ####

##

sbox = 2
ss = range(-sbox, stop=sbox, length=1000)
zs = ss' .- im*range(0, stop=2, length=100)
gr(size=(1100,450))
M  = H_hat.(zs) # F_hat.(zs) .* 
p1 = heatmap(angle.(M), c=:phase, yflip=true, label="")
p2 = heatmap(log10.(abs.(M)), yflip=true, label="")
pl = plot(p1, p2, layout=grid(1,2))
display(pl)

##

vs = range(0, stop=1, length=10000)
u = 100
plot(
    vs,
    [real.(H_hat.(u .- im*vs)) imag.(H_hat.(u .- im*vs))]
)

##

m = 8

pl = plot(
    ss, abs.((gamma.(1/2 .+ im*pi*ss) - gamma_asympt.(1/2 .+ im*pi*ss, m=m)) ./ gamma.(1/2 .- im*pi*ws)) .+ 1e-16,
    yscale=:log10,
    label="",
    ylabel="Error",
    xlabel=L"$s$",
    linewidth=2
)
display(pl)


##

B2 = [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6, -3617/510]
log_gamma_asympt(z; m=8) = (z-1/2)*log(z) - z + log(2pi)/2 + sum(B2[k]/(2k*(2k-1)*z^(2k-1)) for k=1:m)