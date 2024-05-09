using LinearAlgebra, SpecialFunctions, FINUFFT, QuadGK, Plots, Plots.Measures, LaTeXStrings, Printf, BenchmarkTools, Polynomials

Box = NTuple{4, Int64}

function nufht(nu, rs, cs, ws; max_levels=nothing, min_box_dim=100)
    # Wimp expansion currently implemented only for nu = 0
    @assert nu == 0
    @assert issorted(rs) && issorted(ws)

    n = length(ws)
    m = length(rs)

    # initialize output
    gs = zeros(Float64, n)

    if min(n, m) < 100
        # matrix is small enough that direct summation will be faster
        add_dir!(gs, nu, rs, cs, ws)
        return gs
    end

    z = 25
    boxes = generate_boxes(
        rs, ws, z; 
        max_levels=max_levels, min_box_dim=min_box_dim
        )

    for (box_set, add_box!) in zip(boxes, (add_loc!, add_asy!, add_dir!))
        for (i0b, i1b, j0b, j1b) in box_set
            add_box!(
                view(gs, i0b:i1b), nu, 
                rs[j0b:j1b], cs[j0b:j1b], ws[i0b:i1b]
                )
        end
    end

    return gs
end

function generate_boxes(rs, ws, z; max_levels=nothing, min_box_dim=100)
    n, m = length(ws), length(rs)
    if isnothing(max_levels)
        max_levels = floor(Int64, log2(min(n, m) / min_box_dim))
    end

    loc_boxes = Vector{Box}([])
    asy_boxes = Vector{Box}([])
    dir_boxes = Vector{Box}([])

    # find submatrix (i0, i1, j0, j1) which contains both loc and asy regions
    i0 = findfirst(>(z / rs[end]), ws)
    i1 = findfirst(>(z / rs[1]), ws)
    i1 = isnothing(i1) ? n : i1-1

    j0 = findfirst(>(z / ws[end]), rs)
    j1 = findfirst(>(z / ws[1]), rs)
    j1 = isnothing(j1) ? m : j1-1

    if isnothing(i0)
        # all w*r <= z --- use local expansion everywhere
        push!(loc_boxes, (1, n, 1, m))
        return loc_boxes, asy_boxes, dir_boxes
    elseif i1 == 0
        # all w*r > z --- use asymptotic expansion everywhere
        push!(asy_boxes, (1, n, 1, m))
        return loc_boxes, asy_boxes, dir_boxes
    end

    if i0 > 1
        # use local expansion on ws for which all w*r <= z
        push!(loc_boxes, (1, i0-1, 1, m))
    end
    if j0 > 1
        # use local expansion on rs for which all w*r <= z
        push!(loc_boxes, (i0, n, 1, j0-1))
    end
    if i1 < n
        # use asymptotic expansion on ws for which all w*r > z
        push!(asy_boxes, (i1, n, 1, m))
    end
    if j1 < m
        # use asymptotic expansion on rs for which all w*r > z
        push!(asy_boxes, (1, i1-1, j1, m))
    end

    push!(dir_boxes, (i0, i1, j0, j1))
    for _=1:max_levels
        new_dir_boxes = Vector{Box}([])
        for box in dir_boxes
            # split into local, asymptotic, and direct boxes
            i0b, i1b, j0b, j1b = box
            if min(i1b-i0b+1, j1b-j0b+1) > min_box_dim
                (ispl, jspl) = split_box(rs, ws, box, z)
                push!(loc_boxes, (i0b, ispl-1, j0b, jspl-1))
                push!(asy_boxes, (ispl, i1b, jspl, j1b))
                push!(new_dir_boxes, (i0b, ispl-1, jspl, j1b))
                push!(new_dir_boxes, (ispl, i1b, j0b, jspl-1))
            else
                push!(new_dir_boxes, box)
            end
        end
        dir_boxes = new_dir_boxes
    end

    return loc_boxes, asy_boxes, dir_boxes
end

function split_box(rs, ws, box, z)
    # number of equispaced arguments to check
    num_check = 10
    sk = round(Int64, (box[2]-box[1]+1)/num_check)
    # find (i, j) with wi*r_j >= z, i in is, j in js 
    # that maximizes the size of the block A[i:is[end], j:js[end]]
    i  = argmax(
        i -> (box[2]-i+1)*(box[4]-findfirst(>(z / ws[i]), rs)+1), 
        box[1]:sk:box[2]
    )
    j = findfirst(>(z / ws[i]), rs)

    return i, j
end

function number_boxes(n, m, boxes)
    M = fill(NaN, n, m)

    for (box_set, val) in zip(boxes, [2,3,1])
        for box in box_set
            i0b, i1b, j0b, j1b = box
            M[i0b:i1b, j0b:j1b] .= val
        end
    end

    return M
end

function add_tay!(gs, nu, rs, cs, ws; K=100)
    for k=0:K
        gs .+= (-1)^k * (ws/2).^(2k+nu) / (factorial(big(k))*gamma(nu+k+1)) * sum(cs .* rs.^(2k+nu))
    end
end

function add_loc!(gs, nu, rs, cs, ws; K=30)
    @assert nu == 0

    for k=0:K
        gs .+= (-1)^k * (k==0 ? 1 : 2) * besselj.(k, ws*rs[end]/2).^2 * dot(2*ChebyshevT(I[1:k+1, k+1]).(rs/rs[end]).^2 .- 1, cs)
    end
end

a(k, nu) = k==0 ? 1 : prod((4nu^2 .- (1:2:(2k-1)).^2)) / (factorial(big(k))*8^k)

function add_asy!(gs, nu, rs, cs, ws; K=10)
    # initialize temporary vectors
    v1 = zeros(ComplexF64, length(ws))
    v2 = zeros(ComplexF64, length(ws))

    for k=0:K
        v1 .= exp.((-nu/2-1/4)*pi*im) * nufft1d3(
            rs, ComplexF64.(cs .* rs.^(-2k-1/2)), +1, 1e-15, ws
        )
        v2 .= exp.((-nu/2-1/4)*pi*im) * nufft1d3(
            rs, ComplexF64.(cs .* rs.^(-2k-1-1/2)), +1, 1e-15, ws
        ) 
        gs .+= sqrt(2/pi) * (-1)^k * (
              a(2k, nu)   * real.(v1) .* ws.^(-2k-1/2) 
            - a(2k+1, nu) * imag.(v2) .* ws.^(-2k-1-1/2)
            )
    end
end

function add_dir!(gs, nu, rs, cs, ws)
    gs .+= sum(cs' .* besselj.(nu, ws*rs'), dims=2)
end