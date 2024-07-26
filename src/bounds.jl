
# coefficients in Hankel's expansion
a(k, nu) = k==0 ? 1 : Float64(prod(big.(4*nu^2 .- (1:2:(2k-1)).^2)) / (factorial(big(k))*big(8)^k))

function generate_asy_coef_table(nus, max_asy_K)
    Js = 0:(2max_asy_K + 1)
    as = Array{Float64}(undef, length(nus), length(Js))
    for (i, nu) in enumerate(nus)
        for (j, J) in enumerate(Js)
            as[i, j] = a(J, nu)
        end
    end

    return as
end

function asy_error_bound(nu, K, z)
    return sqrt(2/(pi*z)) * (
        abs(a(2K, nu)) / z^(2K) + abs(a(2K+1, nu)) / z^(2K+1)
        )
end

function generate_asy_z_table(nus, asy_Ks, tols)
    zs = Array{Float64}(undef, length(nus), length(tols), length(asy_Ks))
    for (i, nu) in enumerate(nus)
        for (j, tol) in enumerate(tols)
            for (k, K) in enumerate(asy_Ks)
                @printf("computing asymptotic z for ν = %i, tol = %.0e, K = %i...\n", nu, tol, K)

                zs[i, j, k] = newton(
                    z -> asy_error_bound(nu, K, z) - tol,
                    1.0, 1e-8,
                    maxiter=1000, verbose=false
                )
            end
        end
    end

    return zs
end

# exponent in Siegel's bound
beta(p) = log(p) + sqrt(1 - p^2) - log(1 + sqrt(1-p^2))

function wimp_error_bound(nu, K, z)
    @assert z <= 2K + nu

    p   = z / (nu + 2K)
    b_K = beta(p)

    if 2K > nu && z <= 2K - nu
        q   = z / (2K - nu)
        c_K = beta(q)

        return 2*exp(b_K*(nu/2 + K + 1) + c_K*(-nu/2 + K + 1)) / (1 - exp(b_K + c_K))
    else
        return 2*exp(b_K*(nu/2 + K + 1)) / (1 - exp(b_K))
    end
end

function generate_wimp_K_table(nus, asy_Ks, zs, tols)
    wimp_Ks = Array{Int64}(undef, length(nus), length(tols), length(asy_Ks))
    for (i, nu) in enumerate(nus)
        for (j, tol) in enumerate(tols)
            for (k, asy_K) in enumerate(asy_Ks)
                @show nu, tol, asy_K
                @printf("computing Wimp K for ν = %i, tol = %.0e, asymptotic K = %i...\n", nu, tol, asy_K)

                conv = false
                wimp_K = ceil(Int64, (zs[i, j, k] - nu)/2)
                while !conv
                    conv = wimp_error_bound(nu, wimp_K, zs[i, j, k]) < tol
                    wimp_K += 1
                end
                wimp_Ks[i, j, k] = wimp_K - 1
            end
        end
    end

    return wimp_Ks
end

function newton(f, x0, tol; bounds=(-Inf, Inf), maxiter=100, verbose=true)
    i  = 1
    x  = x0
    dx = Inf
    while i < maxiter && abs(dx / x) > tol
        if verbose
            @printf("iteration %i : x = %.6f\n", i, x)
        end

        dx = -f(x) / ForwardDiff.derivative(xp -> f(xp), x)
        dx = min(max(dx, bounds[1]-x), bounds[2]-x)
        x += dx
        i += 1
    end
    
    if i == maxiter
        @warn @sprintf("maxiter %i exceeded with dx = %.2e!\n", maxiter, abs(dx))
    elseif verbose
        @printf("converged with x = %.6f\n", x)
    end

    return x
end