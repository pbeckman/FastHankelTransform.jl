
function generate_tables(path;
                         nus_all=0:0.5:200.0,
                         nus_int=0:200,
                         nltols=4:15,
                         max_K_asy=10)
    @printf("\n--------------------
Generating tables
--------------------\n")

    t = @elapsed begin
        as = generate_asy_coef_table(nus_all, max_K_asy)
        write(path * "asy_coef_table.bin", as)

        zs = generate_z_split_table(nus_int, 1:max_K_asy, nltols)
        write(path * "z_split_table.bin", get_reduced_table(zs, nus_int, nltols))

        Ks = generate_K_loc_table(nus_int, 1:max_K_asy, zs, nltols)
        write(path * "K_loc_table.bin", get_reduced_table(Ks, nus_int, nltols))
    end

    @printf("\n--------------------
Tables succesfully generated! (%.1f s)
--------------------\n", t)
end

# set number of terms in asymptotic expansion based on loose experiments
# to balance effort of asymptotic and local expansions
get_K_asy(nu, nltol) = min(max_K_asy, floor(Int64, abs(nu)/5 + nltol/4 + 1))

function get_reduced_table(table, nus, nltols)
    inds = get_K_asy.(nus, nltols')
    return [table[Tuple(I)...,inds[I]] for I in CartesianIndices(inds)]
end

# coefficients in Hankel's expansion
a(k, nu) = k==0 ? 1 : Float64(prod(big.(4*nu^2 .- (1:2:(2k-1)).^2)) / (factorial(big(k))*big(8)^k))

function generate_asy_coef_table(nus, max_K_asy)
    Js = 0:(2max_K_asy + 1)
    as = Array{Float64}(undef, length(nus), length(Js))
    for (i, nu) in enumerate(nus)
        for (j, J) in enumerate(Js)
            @printf("computing asymptotic coefficient for ν = %.1f, j = %i...\n", nu, J)
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

function generate_z_split_table(nus, K_asys, nltols)
    zs = Array{Float64}(undef, length(nus), length(nltols), length(K_asys))
    for (i, nu) in enumerate(nus)
        for (j, nltol) in enumerate(nltols)
            for (k, K) in enumerate(K_asys)
                @printf("computing asymptotic z for ν = %i, tol = %.0e, K = %i...\n", nu, 10.0^(-nltol), K)

                zs[i, j, k] = newton(
                    z -> asy_error_bound(nu, K, z) - 10.0^(-nltol),
                    1.0, 1e-8,
                    maxiter=1000, verbose=false
                )
            end
        end
    end

    return zs
end

# exponent in Siegel's bound
psi(p) = log(p) + sqrt(1 - p^2) - log(1 + sqrt(1-p^2))

function wimp_error_bound(nu, K, z)
    @assert z <= 2K + nu

    b_K = psi(z / (2K + 2 + nu))

    # determine if Siegel's bound is valid for J_{nu/2-l}
    if 2K - nu > 0 && z / (2K - nu) <= 1
        c_K = psi(z / (2K + 2 - nu))

        return 2*exp(b_K*(nu/2 + K + 1) + c_K*(-nu/2 + K + 1)) / (1 - exp(b_K + c_K))
    else
        return 2*exp(b_K*(nu/2 + K + 1)) / (1 - exp(b_K))
    end
end

function generate_K_loc_table(nus, K_asys, zs, nltols)
    K_locs = Array{Int64}(undef, length(nus), length(nltols), length(K_asys))
    for (i, nu) in enumerate(nus)
        for (j, nltol) in enumerate(nltols)
            for (k, K_asy) in enumerate(K_asys)
                @printf("computing Wimp K for ν = %i, tol = %.0e, asymptotic K = %i...\n", nu, 10.0^(-nltol), K_asy)

                conv = false
                K_loc = ceil(Int64, (zs[i, j, k] - nu)/2)
                while !conv
                    conv = wimp_error_bound(nu, K_loc, zs[i, j, k]) < 10.0^(-nltol)
                    K_loc += 1
                end
                K_locs[i, j, k] = K_loc - 1
            end
        end
    end

    return K_locs
end

function newton(f, x0, tol; bounds=(-Inf, Inf), maxiter=100, verbose=true)
    i  = 1
    x  = x0
    dx = Inf
    while i < maxiter && abs(dx / x) > tol
        if verbose
            @printf("iteration %i : x = %.6f\n", i, x)
        end
        h  = 1e-4
        dx = -f(x) / ((f(x+h) - f(x-h)) / (2h))
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
