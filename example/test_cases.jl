
function test_case(case, n, m)
    pts = nothing
    if case == :one
        rs = [1.0]
        m  = 1
        ws = collect(range(0, stop=100, length=n))
    elseif case == :exp
        ws = 10 .^ range(log10(0.25), stop=2, length=n)
        rs = 10 .^ range(log10(0.25), stop=2, length=m)
    elseif case == :roots
        ws = FastGaussQuadrature.approx_besselroots(nu, n+1)
        rs = ws[1:end-1] / ws[end] 
        ws = ws[1:end-1]
    elseif case == :twodimrandom
        npt    = round(Int64, (sqrt(8n-7) + 1)/2)
        n      = Int64(npt*(npt-1)/2 + 1)
        pts    = rand(2, npt)
        ws     = sort(vec(vcat(
            norm.([pts[:,i] - pts[:,j] for i=1:npt for j=i+1:npt]), 0
            )))
        aa = 0
        bb = m / (2minimum(ws[ws .> 0]))
        rs, cs = gausslegendre(m)
        rs  .= (rs .+ 1) * (bb-aa)/2 .+ aa
        L    = 0.1
        cs .*= (bb-aa)/2 * exp.(-(L*rs).^2) .* rs
    elseif case == :loc
        ws = collect(range(0, stop=2, length=n))
        rs = collect(range(0, stop=2, length=m))
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

    cs = ones(m)

    return rs, cs, ws, n, m, pts
end