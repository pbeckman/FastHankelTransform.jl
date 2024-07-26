
function add_dir!(gs, nu, rs, cs, ws)
    @timeit TIMER "Direct" begin
        for k in eachindex(ws)
            @inbounds begin
                wk = ws[k]
                @simd for j in eachindex(rs)
                    gs[k] += cs[j] * besselj(nu, wk*rs[j]) 
                end
            end
        end
    end
end

function add_loc!(gs, nu, rs, cs, ws; K=30)
    # Wimp expansion only works for integer nu
    @assert isinteger(nu)

    @timeit TIMER "Local" begin
        # TODO (pb 6/25/2024): use recurrences for Chebyshev and Bessel evals
        for k=0:K
            if nu == 0
                gs .+= (-1)^k * (k==0 ? 1 : 2) * besselj.(k, ws*rs[end]/2).^2 * dot(2*ChebyshevT(I[1:k+1, k+1]).(rs/rs[end]).^2 .- 1, cs)
            elseif iseven(nu)
                gs .+= (k==0 ? 1 : 2) * besselj.(div(nu,2) + k, ws*rs[end]/2) .* besselj.(div(nu,2) - k, ws*rs[end]/2) * dot(ChebyshevT(I[1:2k+1, 2k+1]).(rs/rs[end]), cs)
            else
                gs .+= 2 * besselj.(div(nu,2) + k + 1, ws*rs[end]/2) .* besselj.(div(nu,2) - k, ws*rs[end]/2) * dot(ChebyshevT(I[1:2k+2, 2k+2]).(rs/rs[end]), cs)
            end
        end
    end
end

function add_asy!(gs, nu, rs, cs, ws; 
    K=5, in_buffer=nothing, out_buffer=nothing)
    @timeit TIMER "Asymptotic" begin
        # initialize temporary buffers for NUFFTs
        if isnothing(in_buffer)
            in_buffer = zeros(ComplexF64, length(rs))
        end
        if isnothing(out_buffer)
            out_buffer = zeros(ComplexF64, length(ws))
        end

        for k=0:K
            @timeit TIMER "NUFFT" begin
                in_buffer .= cs .* rs.^(-2k-1/2)
                nufft1d3!(
                    rs, in_buffer, +1, NUFHT_TOL[], 
                    ws, out_buffer
                )
            end
            @timeit TIMER "Add NUFFT to output" begin
                out_buffer .*= exp((-nu/2-1/4)*pi*im)
                flbuf = reinterpret(Float64, out_buffer)
                buf   = @view flbuf[1:2:end-1] # take real part
                buf .*= sqrt(2/pi) * (-1)^k * NUFHT_ASY_COEF[][2k+1]
                buf .*= ws.^(-2k-1/2)
                gs  .+= buf
            end

            @timeit TIMER "NUFFT" begin 
                in_buffer .= cs .* rs.^(-2k-1-1/2)
                nufft1d3!(
                    rs, in_buffer, +1, NUFHT_TOL[], 
                    ws, out_buffer
                ) 
            end
            @timeit TIMER "Add NUFFT to output" begin
                out_buffer .*= exp((-nu/2-1/4)*pi*im)
                flbuf = reinterpret(Float64, out_buffer)
                buf   = @view flbuf[2:2:end] # take imag part
                buf .*= sqrt(2/pi) * (-1)^k * NUFHT_ASY_COEF[][2k+2]
                buf .*= ws.^(-2k-1-1/2)
                gs  .+= buf
            end
        end
    end
end

##########
# Everything below is experimental
##########

# coefficient ratios b_k = a_{2k} / a_{2k-2}, c_k = a_{2k+1} / a_{2k-1}
b(k, nu) = k==0 ? 1 : (4*nu^2 - (4k-3)^2)*(4*nu^2 - (4k-1)^2) / (2k*(2k-1)*8^2)
c(k, nu) = k==0 ? 1 : (4*nu^2 - (4k-1)^2)*(4*nu^2 - (4k+1)^2) / ((2k+1)*2k*8^2)

function add_asy_horner!(gs, nu, rs, cs, ws; K=5, buffer=nothing)
    @timeit TIMER "Asymptotic Horner" begin
        if isnothing(buffer)
            # initialize temporary buffer for NUFFTs
            buffer = zeros(ComplexF64, length(ws))
        end

        for k=0:K
            @timeit TIMER "NUFFT" nufft1d3!(
                rs, ComplexF64.(cs .* rs.^(-2k-1/2)), +1, 1e-15, 
                ws, buffer
            )
            @timeit TIMER "Add NUFFT to output" begin
                buffer .*= exp((-nu/2-1/4)*pi*im)
                gs .+= sqrt(2/pi) * (-1)^k * a(2k, nu) * real.(buffer) .* ws.^(-2k-1/2)
            end
        end

        for k=0:K
            @timeit TIMER "NUFFT" nufft1d3!(
                rs, ComplexF64.(cs .* rs.^(-2k-1-1/2)), +1, 1e-15, 
                ws, buffer
            ) 
            @timeit TIMER "Add NUFFT to output" begin
                buffer .*= exp((-nu/2-1/4)*pi*im)
                gs .-= sqrt(2/pi) * (-1)^k * a(2k+1, nu) * imag.(buffer) .* ws.^(-2k-1-1/2)
            end
        end
    end
end

function add_taylor!(gs, nu, rs, cs, ws; K=30)
    @timeit TIMER "Taylor" begin
        for k=0:K
            gs .+= (-1)^k * (ws/2).^(2k+nu) / (factorial(big(k))*gamma(nu+k+1)) * sum(cs .* rs.^(2k+nu))
        end
    end
end

function U(k, p)
    if k == 0
        return 1
    elseif k == 1
        return (3p - 5p^3) / 24
    elseif k == 2
        return  (81p^2 - 462p^4 + 385p^6) / 1152
    elseif k == 3
        return (30375p^3 - 369603p^5 + 765765p^7 - 425425p^9) / 414720
    else
        error("U(k, p) only implemented for k ≤ 3.")
    end
end

asy_beta(z; K=3) = pi/2 - Polynomial(
    [0, 1, 0, 1/6, 0, 3/40, 0, 5/112, 0, 35/1152, 0, 63/2816, 0, 231/13312, 0, 143/10240][1:K]
    )(1/z)
asy_tanbeta(z; K=3) = z - Polynomial(
    [0, 1/2, 0, 1/8, 0, 1/16, 0, 5/128, 0, 7/256, 0, 21/1024, 0, 33/2048, 0, 429/32768][1:K]
    )(1/z)
asy_cotbeta(z; K=3) = Polynomial(
    [0, 1, 0, 1/2, 0, 3/8, 0, 5/16, 0, 35/128, 0, 63/256, 0, 231/1024, 0, 429/2048, 0, 6435/32768][1:K]
    )(1/z)

function add_debye!(gs, nu, rs, cs, ws; K=3, large_z=false, Kz=3)
    @timeit TIMER "Debye" begin
        for k=0:K
            as = asech.(ws[ws .<= nu]*rs' / nu)
            view(gs, ws .<= nu) .+= exp.(nu*(tanh.(as) - as)) ./ sqrt.(2pi*nu*tanh.(as)) .* U.(k, coth.(as)) / nu^k * cs
        end

        for k=0:div(K, 2)
            zs  = ws[ws .> nu]*rs' / nu
            if large_z
                bs  = asy_beta.(zs, K=Kz)
                xis = nu * (asy_tanbeta.(zs, K=Kz) .- bs) .- pi/4
                view(gs, ws .> nu) .+= sqrt.(2 ./ (pi*nu*asy_tanbeta.(zs, K=Kz))) .* (
                     cos.(xis).*U.(2k,   im*asy_cotbeta.(zs, K=Kz)) ./ nu^(2k)
                .-im*sin.(xis).*U.(2k+1, im*asy_cotbeta.(zs, K=Kz)) ./ nu^(2k+1)
                ) * cs
            else
                bs  = asec.(zs)
                xis = nu * (tan.(bs) .- bs) .- pi/4
                view(gs, ws .> nu) .+= sqrt.(2 ./ (pi*nu*tan.(bs))) .* (
                            cos.(xis) .* U.(2k,   im*cot.(bs)) ./ nu^(2k)
                    .- im * sin.(xis) .* U.(2k+1, im*cot.(bs)) ./ nu^(2k+1)
                ) * cs
            end
        end
    end
end