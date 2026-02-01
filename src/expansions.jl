
function add_dir!(gs, nu, rs, cs, ws)
    for j in eachindex(ws)
        @inbounds begin
            wj = ws[j]
            @simd for k in eachindex(rs)
                gs[j] += cs[k] * besselj(nu, wj*rs[k]) 
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

function add_asy!(gs, nu, rs, cs, ws, asy_coef; 
    K=5, tol=1e-15, in_buffer=nothing, 
    out_buffer=nothing, real_buffer_1=nothing, real_buffer_2=nothing)
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
        # write cs .* rs.^(-2l-1/2) to buffer
        in_buffer .= cs .* (rs.^(-2l-1)) .* sqrt.(rs)
        nufft1d3!(rs, in_buffer, +1, tol, ws, out_buffer)
        out_buffer .*= cispi(-nu/2-1/4)
        @inbounds @simd for j in eachindex(out_buffer)
            # write real part of NUFFT output to buffer 1
            real_buffer_1[j] = real(out_buffer[j])
            # write ws.^(-2l-1/2) to buffer 2
            real_buffer_2[j] = (ws[j]^(-2l-1))*sqrt(ws[j])
        end
        # multiply by coefficient and do diagonal scaling
        real_buffer_1 .*= sqrt(2/pi) * (-1)^l * asy_coef[2l+1]
        real_buffer_1 .*= real_buffer_2
        gs            .+= real_buffer_1

        # write cs .* rs.^(-2l-1-1/2) to buffer
        in_buffer .= cs .* (rs.^(-2l-2)) .* sqrt.(rs)
        nufft1d3!(rs, in_buffer, +1, tol, ws, out_buffer) 
        out_buffer .*= cispi(-nu/2-1/4)
        int_exp = -2l-2
        @inbounds @simd for j in eachindex(out_buffer)
            # write imaginary part of NUFFT output to buffer 1
            real_buffer_1[j] = imag(out_buffer[j])
            # write ws.^(-2l-1-1/2) to buffer 2
            real_buffer_2[j] = (ws[j]^int_exp)*sqrt(ws[j])
        end
        # multiply by coefficient and do diagonal scaling
        real_buffer_1 .*= sqrt(2/pi) * (-1)^l * asy_coef[2l+2]
        real_buffer_1 .*= real_buffer_2
        gs            .-= real_buffer_1
    end
end