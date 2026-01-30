module FastHankelTransform
    using LinearAlgebra, FINUFFT, Printf, TimerOutputs
    import Bessels: besselj, besselj!

    export nufht, nufht!

    const TIMER = TimerOutput()

    const nu_max    = 200
    const nus_all   = 0:0.5:nu_max 
    const nus_int   = 0:nu_max
    const nltols    = 4:15 # negative log tolerances
    const max_K_asy = 10

    include("bounds.jl")
    
    # path to tables
    path = joinpath(@__DIR__, "..", "tables/")
    if !ispath(path)
        mkdir(path)
        generate_tables(path)
    end

    # read tables into constants
    tmp = Array{Float64}(undef, length(nus_all), 2max_K_asy+2)
    open(path * "asy_coef_table.bin") do file
        read!(file, tmp)
    end
    const ASY_COEF_TABLE = copy(tmp)

    tmp = Array{Float64}(undef, length(nus_int), length(nltols))
    open(path * "z_split_table.bin") do file
        read!(file, tmp)
    end
    const Z_SPLIT_TABLE = copy(tmp)

    tmp = Array{Int64}(undef, length(nus_int), length(nltols))
    open(path * "K_loc_table.bin") do file
        read!(file, tmp)
    end
    const K_LOC_TABLE = copy(tmp)

    include("expansions.jl")
    include("nufht.jl")
end