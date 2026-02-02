module FastHankelTransform
    using LinearAlgebra, FINUFFT, Printf
    import Bessels: besselj, besselj!

    export nufht, nufht!

    const nu_max    = 200
    const nus_all   = 0:0.5:nu_max 
    const nus_int   = 0:nu_max
    const nltols    = 4:15 # negative log tolerances
    const max_K_asy = 10

    include("bounds.jl")
   
    # read (or generate) the pre-computed tables.
    const (ASY_COEF_TABLE, Z_SPLIT_TABLE, K_LOC_TABLE) = let path = joinpath(@__DIR__, "..", "tables/")
      # regenerate tables if they aren't given.
      if !ispath(path)
          mkdir(path)
          generate_tables(path)
      end
      # otherwise, load them in.
      asy_coef_table = Array{Float64}(undef, length(nus_all), 2max_K_asy+2)
      open(path * "asy_coef_table.bin") do file
          read!(file, asy_coef_table)
      end
      z_split_table = Array{Float64}(undef, length(nus_int), length(nltols))
      open(path * "z_split_table.bin") do file
          read!(file, z_split_table)
      end
      k_loc_table = Array{Int64}(undef, length(nus_int), length(nltols))
      open(path * "K_loc_table.bin") do file
          read!(file, k_loc_table)
      end
      (asy_coef_table, z_split_table, k_loc_table)
    end

    include("expansions.jl")
    include("nufht.jl")
end
