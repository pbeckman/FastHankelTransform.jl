using LinearAlgebra, SpecialFunctions, FINUFFT, QuadGK, Plots, Plots.Measures, LaTeXStrings, Printf

gr(size=(1000, 300))
default(margin=5mm)

nu  = 0
tau = 0.001
rk  = 0.5
c   = 1/sqrt(2pi*tau^2)

F(t)   = c * exp(-t) * exp(-(t + log(rk)).^2 / (2tau^2))
H(t)   = exp(t) * besselj(nu, exp(t))
FcH(t) = quadgk(u -> F(u)*H(t-u), -log(rk)-10tau, -log(rk)+10tau)

ts = range(-2, stop=4, length=1000)

F_hat(s) = c*sqrt(2pi)*tau*rk * exp(-tau^2/2*(2pi*s - im)^2 + 2pi*log(rk)*im*s)
B2 = [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6, -3617/510]
log_gamma_asympt(z; m=8) = (z-1/2)*log(z) - z + log(2pi)/2 + sum(B2[k]/(2k*(2k-1)*z^(2k-1)) for k=1:m)
function H_hat(s) 
    if abs(s) < 100 
        out = gamma(1/2 - im*pi*s)/gamma(1/2 + im*pi*s)
    else
        out = exp(
            log_gamma_asympt(1/2 - im*pi*s) - 
            log_gamma_asympt(1/2 + im*pi*s)
            )
    end
    return 2^(-2pi*im*s) * out
end

tol = 1e-6

S = 1
while abs(F_hat(S)) > tol
    S *= 2
end
ss = range(-S/2, stop=S/2, length=1000)
p2 = plot(
    ss,
    [real.(F_hat.(ss) .* H_hat.(ss)) imag.(F_hat.(ss) .* H_hat.(ss))], 
    labels=[L"$Re(\widehat{F}\widehat{H})$" L"$Im(\widehat{F}\widehat{H})$"], 
    xlabel="s",
    color=[:red :blue], line=2
    )
display(p2)

m = 1000
FcH_fourier = fill(ComplexF64(eps()), length(ts))
err         = fill(ComplexF64(Inf),   length(ts))
while maximum(abs.(err ./ FcH_fourier)) > tol
    m   *= 2
    h    = 2S / (m-1)
    nds  = collect((-S):h:S)
    err .= h*nufft1d3(
        nds, 
        F_hat.(nds) .* H_hat.(nds), 
        +1, 1e-15, 
        2pi*collect(ts)
        ) - FcH_fourier
    FcH_fourier .+= err
end
FcH_quadgk = getindex.(FcH.(ts),1)

@printf(
    "relative 2-norm error in F*H : %.2e\n", 
    norm(FcH_fourier - FcH_quadgk) / norm(FcH_quadgk)
    )

p1 = plot(
    ts, 
    [F.(ts) H.(ts)], 
    labels=[L"F" L"H" L"(F*H)"], 
    xlabel=L"t", line=2,
    legend=:topleft
    )
plot!(p1,
    ts,
    real.(FcH_quadgk), 
    label=L"(F*H)",
    line=2
)
plot!(p1,
    ts,
    real.(FcH_fourier), 
    label=L"\mathcal{F}^{-1}(\widehat{F}\widehat{H})",
    line=(2, :dash)
)
display(p1)

bessel_evals  = besselj.(nu, exp.(ts)*rk) * rk^2 # why this squared?
fourier_evals = exp.(-ts) .* real.(FcH_fourier)
p3 = plot(
    exp.(ts),
    bessel_evals,
    label=L"J_\nu(\omega r) r",
    line=2,
    xlabel=L"\omega",
    legend=:topright
    )
plot!(p3,
    exp.(ts),
    fourier_evals,
    label=L"\frac{1}{\omega}\mathcal{F}^{-1}(\widehat{F}\widehat{H})(\log\omega)",
    line=(2, :dash)
)
display(p3)

p4 = plot(
    exp.(ts),
    abs.((bessel_evals - fourier_evals) ./ bessel_evals),
    label=L"relative error in $J_\nu(\omega r)$",
    yscale=:log10,
    line=(2, :purple),
    xlabel=L"\omega",
    legend=:topright
    )
display(p4)

##

sbox = 100
ss = range(-sbox, stop=sbox, length=1000)
gr(size=(1100,450))
M  = F_hat.(ss' .+ im*ss) # .* H_hat.(ss' .+ im*ss)
p1 = heatmap(angle.(M), c=:phase, label="")
p2 = heatmap(log10.(abs.(M)), label="")
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

ks  = 0:10
res = (-1).^ks .* 2.0 .^ (-(2ks .+ nu .+ 1)) ./ (gamma.(ks .+ 1) .* gamma.(nu .+ ks .+ 1))