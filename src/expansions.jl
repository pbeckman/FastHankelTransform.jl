
function add_dir!(gs, nu, rs, cs, ws)
    @timeit TIMER "Direct" begin
        for j in eachindex(ws)
            @inbounds begin
                wj = ws[j]
                @simd for k in eachindex(rs)
                    gs[j] += cs[k] * besselj(nu, wj*rs[k]) 
                end
            end
        end
    end
end

function add_loc!(gs, nu, rs, cs, ws; 
    K=30, cheb_buffer=nothing, bessel_buffer_1=nothing, bessel_buffer_2=nothing)
    # Wimp expansion only works for integer nu
    @assert isinteger(nu)

    n = length(ws)
    m = length(rs)

    # center index of Wimp expansion
    l0 = div(nu,2)

    @timeit TIMER "Local" begin
        # initialize temporary buffers for coefficients and Bessel evals
        if isnothing(cheb_buffer)
            cheb_buffer = zeros(Float64, K+1)
        else
            fill!(cheb_buffer, 0.0)
        end
        if isnothing(bessel_buffer_1)
            bessel_buffer_1 = zeros(Float64, K+1)
        else
            fill!(bessel_buffer_1, 0.0)
        end
        if nu != 0 
            if isnothing(bessel_buffer_2)
                bessel_buffer_2 = zeros(Float64, l0+K+1+isodd(nu))
            else
                fill!(bessel_buffer_2, 0.0)
            end
        end

        for l=0:K
            @inbounds for k in eachindex(rs)
                # evaluate Chebyshev polynomials
                if nu == 0
                    cheb_buffer[l+1] += (l==0 ? 1 : 2) * (2cos(l*acos(rs[k]/rs[end]))^2 - 1) * cs[k]
                elseif iseven(nu)
                    cheb_buffer[l+1] += (l==0 ? 1 : 2) * cos(2l*acos(rs[k]/rs[end])) * cs[k]
                else
                    cheb_buffer[l+1] += 2 * cos((2l+1)*acos(rs[k]/rs[end])) * cs[k]
                end
            end
        end

        @inbounds for j in eachindex(ws)
            if nu == 0
                # compute all necessary Bessel functions evals
                besselj!(bessel_buffer_1, 0:K, ws[j]*rs[end]/2)
                # square evals since orders l and -l have same magnitude 
                bessel_buffer_1 .*= bessel_buffer_1
                # use J_{-l} = (-1)^l J_l for negative orders
                view(bessel_buffer_1, 2:2:(K+1)) .*= -1
            else
                # compute all necessary Bessel functions evals
                besselj!(bessel_buffer_2, 0:(l0+K+isodd(nu)), ws[j]*rs[end]/2)
                # copy orders l0...l0+K to buffer (starting index is 1 if odd)
                bessel_buffer_1 .= view(bessel_buffer_2, l0+1+isodd(nu):length(bessel_buffer_2))
                # multiply by orders l0...0
                view(bessel_buffer_1, l0+1:-1:1) .*= view(bessel_buffer_2, 1:l0+1)
                # multiply by orders 1...K-l0
                view(bessel_buffer_1, l0+2:(K+1)) .*= view(bessel_buffer_2, 2:K-l0+1)
                # use J_{-n} = (-1)^n J_n for negative orders
                view(bessel_buffer_1, l0+2:2:(K+1)) .*= -1
            end
            gs[j] += dot(bessel_buffer_1, cheb_buffer)
        end
    end
end

function add_asy!(gs, nu, rs, cs, ws; 
    K=5, in_buffer=nothing, 
    out_buffer=nothing, real_buffer_1=nothing, real_buffer_2=nothing)
    @timeit TIMER "Asymptotic" begin
        # initialize temporary buffers for NUFFTs
        if isnothing(in_buffer)
            in_buffer = zeros(ComplexF64, length(rs))
        end
        if isnothing(out_buffer)
            out_buffer = zeros(ComplexF64, length(ws))
        end
        if isnothing(real_buffer_1)
            real_buffer_1 = zeros(Float64, length(ws))
        end
        if isnothing(real_buffer_2)
            real_buffer_2 = zeros(Float64, length(ws))
        end

        for l=0:K
            @timeit TIMER "Set up NUFFT input" begin
                # write cs .* rs.^(-2l-1/2) to buffer
                in_buffer .= cs .* (rs.^(-2l-1)) .* sqrt.(rs)
            end
            @timeit TIMER "NUFFT" begin
                nufft1d3!(
                    rs, in_buffer, +1, NUFHT_TOL[], 
                    ws, out_buffer
                )
            end
            @timeit TIMER "Add NUFFT to output" begin
                out_buffer .*= cispi(-nu/2-1/4)
                @inbounds @simd for j in eachindex(out_buffer)
                    # write real part of NUFFT output to buffer 1
                    real_buffer_1[j] = real(out_buffer[j])
                    # write ws.^(-2l-1/2) to buffer 2
                    real_buffer_2[j] = (ws[j]^(-2l-1))*sqrt(ws[j])
                end
                # multiply by coefficient and do diagonal scaling
                real_buffer_1 .*= sqrt(2/pi) * (-1)^l * NUFHT_ASY_COEF[][2l+1]
                real_buffer_1 .*= real_buffer_2
                gs            .+= real_buffer_1
            end

            @timeit TIMER "Set up NUFFT input" begin
                # write cs .* rs.^(-2l-1-1/2) to buffer
                in_buffer .= cs .* (rs.^(-2l-2)) .* sqrt.(rs)
            end
            @timeit TIMER "NUFFT" begin
                nufft1d3!(
                    rs, in_buffer, +1, NUFHT_TOL[], 
                    ws, out_buffer
                ) 
            end
            @timeit TIMER "Add NUFFT to output" begin
                out_buffer .*= cispi(-nu/2-1/4)
                int_exp = -2l-2
                @inbounds @simd for j in eachindex(out_buffer)
                    # write imaginary part of NUFFT output to buffer 1
                    real_buffer_1[j] = imag(out_buffer[j])
                    # write ws.^(-2l-1-1/2) to buffer 2
                    real_buffer_2[j] = (ws[j]^int_exp)*sqrt(ws[j])
                end
                # multiply by coefficient and do diagonal scaling
                real_buffer_1 .*= sqrt(2/pi) * (-1)^l * NUFHT_ASY_COEF[][2l+2]
                real_buffer_1 .*= real_buffer_2
                gs            .-= real_buffer_1
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