module FastHankelTransform

    using LinearAlgebra, FINUFFT, ForwardDiff
    using JLD, Printf, TimerOutputs
    import Bessels: besselj, besselj!

    export nufht, nufht!

    const TIMER = TimerOutput()

    global const NUFHT_NU       = Ref{Rational{Int64}}(-1)
    global const NUFHT_TOL      = Ref{Float64}(NaN)
    global const NUFHT_Z_SPLIT  = Ref{Float64}(NaN)
    global const NUFHT_ASY_K    = Ref{Int64}(-1)
    global const NUFHT_ASY_COEF = Ref{Vector{Float64}}([])
    global const NUFHT_LOC_K    = Ref{Int64}(-1)

    include("expansions.jl")
    include("bounds.jl")
    include("nufht.jl")
    
end