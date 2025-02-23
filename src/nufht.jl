
Box   = NTuple{4, Int64}
Boxes = Tuple{Vector{Box}, Vector{Box}, Vector{Box}}

function nufht(nu, rs, cs, ws; 
        tol=1e-8, max_levels=nothing, min_dim_prod=10_000, 
        z_split=nothing, K_asy=nothing, K_loc=nothing) 
    # initialize output
    gs = zeros(Float64, length(ws))

    return nufht!(gs, nu, rs, cs, ws; 
    tol=tol, max_levels=max_levels, min_dim_prod=min_dim_prod, 
    z_split=z_split, K_asy=K_asy, K_loc=K_loc)
end

function nufht!(gs, nu, rs, cs, ws; 
        tol=1e-8, max_levels=nothing, min_dim_prod=10_000, 
        z_split=nothing, K_asy=nothing, K_loc=nothing)
    
    @assert issorted(rs) && issorted(ws)
    @assert length(rs) == length(cs)
    @assert length(gs) == length(ws)

    if any([nu, tol, z_split, K_asy, K_loc] .!= [NUFHT_NU, NUFHT_TOL, NUFHT_Z_SPLIT, NUFHT_ASY_K, NUFHT_LOC_K])
        # different parameters are being used
        # set global variables according to error analysis
        setup_nufht!(nu, tol, z_split=z_split, K_asy=K_asy, K_loc=K_loc)
    end

    n = length(ws)
    m = length(rs)

    if n*m < min_dim_prod
        # matrix is small enough that direct summation will be faster
        add_dir!(gs, nu, rs, cs, ws)
        return gs
    elseif !isinteger(nu)
        # asymptotic expansion is analytic for half integer orders
        add_asy!(gs, nu, rs, cs, ws, K=NUFHT_ASY_K[])
        return gs
    end

    @timeit TIMER "Generate boxes" begin
        boxes = generate_boxes(
            rs, ws, z_split=NUFHT_Z_SPLIT[]; 
            max_levels=max_levels, min_dim_prod=min_dim_prod
            )
    end

    @timeit TIMER "Initialize buffers" begin
        # initialize temporary buffers
        in_buffer  = zeros(ComplexF64, m)
        out_buffer = zeros(ComplexF64, n)
        real_buffer_1 = zeros(Float64, n)
        real_buffer_2 = zeros(Float64, n)
        cheb_buffer     = zeros(Float64, NUFHT_LOC_K[]+1)
        bessel_buffer_1 = zeros(Float64, NUFHT_LOC_K[]+1)
        bessel_buffer_2 = zeros(
            Float64, NUFHT_LOC_K[]+1+div(abs(nu),2)+isodd(nu)
            )
    end

    # add contributions of all boxes
    for (box_set, add_box!) in zip(boxes, (
        (gs, nu, rs, cs, ws) -> add_loc!(
            gs, nu, rs, cs, ws, K=NUFHT_LOC_K[],
            cheb_buffer=cheb_buffer, 
            bessel_buffer_1=bessel_buffer_1, 
            bessel_buffer_2=bessel_buffer_2
            ), 
        (gs, nu, rs, cs, ws) -> add_asy!(
            gs, nu, rs, cs, ws, K=NUFHT_ASY_K[], 
            # TODO (pb 2/10/24): ask FINUFFT.jl to accept SubArrays
            # in_buffer=view(in_buffer, 1:length(rs)),
            # out_buffer=view(out_buffer, 1:length(ws)),
            real_buffer_1=view(real_buffer_1, 1:length(ws)),
            real_buffer_2=view(real_buffer_2, 1:length(ws))
            ),
        add_dir!
        ))
        for (i0b, i1b, j0b, j1b) in box_set
            add_box!(
                view(gs, i0b:i1b), abs(nu),
                rs[j0b:j1b], # view(rs, j0b:j1b), 
                view(cs, j0b:j1b), 
                ws[i0b:i1b]  # view(ws, i0b:i1b)
                )
        end
    end

    if nu < 0 && isodd(nu)
        # use J_{-n} = (-1)^n J_n for negative orders
        gs .*= -1
    end

    return gs
end

function setup_nufht!(nu, tol; z_split=nothing, K_asy=nothing, K_loc=nothing)
    if tol < 1e-15
        error("cannot set NUFHT tolerance below 1e-15")
    end
    if !isinteger(2nu) || (isinteger(nu) && nu > 200) || (!isinteger(nu) && isinteger(2nu) && nu > 19/2)
        error("only NUFHT with integer orders ν = 0,±1,...,±200 and half-integer orders ν = 1/2,3/2,...,19/2 are implemented") 
    end

    # path to tables
    path = join(split(pathof(FastHankelTransform), '/')[1:end-2], '/') * "/tables/"

    # generate tables if they don't yet exist
    if !isdir(path)
        generate_tables()
    end

    # set order
    i = abs(nu) + 1
    global NUFHT_NU[]       = nu
    global NUFHT_ASY_COEF[] = load(path * "asy_a_table.jld")["as"][Int64(2abs(nu) + 1), :]

    # set tolerance
    j = max(1, ceil(Int64, -log10(tol) - 3))
    global NUFHT_TOL[] = tol

    if isinteger(nu)
        # set number of terms in asymptotic expansion based on loose experiments
        # to balance effort of asymptotic and local expansions
        k = isnothing(K_asy) ? min(10, floor(Int64, abs(nu)/5 + log10(1/tol)/4 + 1)) : K_asy
        global NUFHT_ASY_K[] = k

        global NUFHT_Z_SPLIT[] = isnothing(z_split) ? load(path * "asy_z_table.jld")["zs"][i, j, k] : z_split
        global NUFHT_LOC_K[]   = isnothing(K_loc) ? load(path * "wimp_K_table.jld")["Ks"][i, j, k] : K_loc
    else
        # set number of Hankel expansion terms to give exact formula
        global NUFHT_ASY_K[] = Int64(abs(nu) - 1/2)

        # set unused constants to defaults
        global NUFHT_Z_SPLIT[] = NaN
        global NUFHT_LOC_K[]   = -1
    end
end

function generate_boxes(rs, ws; 
    z_split=NUFHT_Z_SPLIT[], max_levels=nothing, min_dim_prod=10_000)
    n, m = length(ws), length(rs)
    if isnothing(max_levels)
        max_levels = floor(Int64, log2(min(n, m)^2 / min_dim_prod))
    end

    loc_boxes = Vector{Box}([])
    asy_boxes = Vector{Box}([])
    dir_boxes = Vector{Box}([])

    # find submatrix (i0, i1, j0, j1) which contains both loc and asy regions
    i0 = findfirst(>(z_split / rs[end]), ws)
    if m == 1
        i1 = i0-1
    else
        i1 = findfirst(>(z_split / rs[1]), ws)
        i1 = isnothing(i1) ? n : i1-1
    end

    j0 = findfirst(>(z_split / ws[end]), rs)
    if n == 1
        j1 = j0-1
    else
        j1 = findfirst(>(z_split / ws[1]), rs)
        j1 = isnothing(j1) ? m : j1-1
    end

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
        push!(asy_boxes, (i1+1, n, 1, m))
    end
    if j1 < m
        # use asymptotic expansion on rs for which all w*r > z
        push!(asy_boxes, (1, i1, j1+1, m))
    end

    if (m == 1) || (n == 1)
        # if testing a single source or target, there are no direct boxes
        return loc_boxes, asy_boxes, dir_boxes
    end

    # list all boxes which are not fully local or fully asymptotic as direct
    push!(dir_boxes, (i0, i1, j0, j1))

    # hierarchically split each direct box into local, asymptotic, and direct
    for _=1:max_levels
        new_dir_boxes = Vector{Box}([])
        for box in dir_boxes
            i0b, i1b, j0b, j1b = box
            if (i1b-i0b+1)*(j1b-j0b+1) > 4*min_dim_prod
                (ispl, jspl) = split_box(rs, ws, box, z_split)
                if ispl > i0b
                    push!(loc_boxes, (i0b, ispl-1, j0b, jspl-1))
                    push!(new_dir_boxes, (i0b, ispl-1, jspl, j1b))
                end
                # TODO (pb 6/27/2024): what about when jspl = j0b?
                push!(asy_boxes, (ispl, i1b, jspl, j1b))
                push!(new_dir_boxes, (ispl, i1b, j0b, jspl-1))
            else
                push!(new_dir_boxes, box)
            end
        end
        dir_boxes = new_dir_boxes
    end

    # move all sufficiently small boxes to direct
    new_loc_boxes = Vector{Box}([])
    new_asy_boxes = Vector{Box}([])
    for (box_set, new_box_set, criterion) in zip(
        [loc_boxes, asy_boxes], 
        [new_loc_boxes, new_asy_boxes], 
        [(d1,d2)->(min(d1,d2) < NUFHT_LOC_K[]), (d1,d2)->(d1*d2 < min_dim_prod)]
        )
        for box in box_set
            d1, d2 = box[2]-box[1]+1, box[4]-box[3]+1
            if criterion(d1, d2)
                push!(dir_boxes, box)
            else
                push!(new_box_set, box)
            end
        end
    end
    loc_boxes = new_loc_boxes
    asy_boxes = new_asy_boxes

    return loc_boxes, asy_boxes, dir_boxes
end

function split_box(rs, ws, box, z)
    @timeit TIMER "Split box" begin
        # number of equispaced arguments to check
        num_check = 10
        # find (i, j) with wi*r_j >= z, i in is, j in js 
        # that maximizes heuristic splitting objective
        i = argmax(
            i -> splitting_objective(rs, ws, box, z, i), 
            unique(round.(Int64, range(box[1], stop=box[2], length=num_check)))
        )
        j = findfirst(>(z / ws[i]), rs)

        return i, j
    end
end

function splitting_objective(rs, ws, box, z, i)
    # the total number of entries for which fast expansions can be used
    j = findfirst(>(z / ws[i]), rs)
    return (i-box[1]+1)*(j-box[3]+1) + (box[2]-i+1)*(box[4]-j+1)
end