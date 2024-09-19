using FINUFFT, BenchmarkTools

tol = 1e-12

ns = 2 .^ (2:14)
ms = 2 .^ (2:14)

nufft_timings = Array{Float64}(undef, length(ns), length(ms))
for (i, n) in enumerate(ns)
    for (j, m) in enumerate(ms)
        ws = rand(n)
        rs = rand(m)
        out_buffer = Vector{ComplexF64}(undef, n)
        in_buffer  = randn(m) .+ im*randn(m)
        println("Timing NUFFT n = $n, m = $m...")
        nufft_timings[i, j] = @belapsed nufft1d3!(
                    $rs, $in_buffer, +1, $tol, 
                    $ws, $out_buffer
                )
    end
end

##

direct_timings = Array{Float64}(undef, length(ns), length(ms))
for (i, n) in enumerate(ns)
    for (j, m) in enumerate(ms)
        ws = rand(n)
        rs = rand(m)
        out_buffer = Vector{ComplexF64}(undef, n)
        in_buffer  = randn(m) .+ im*randn(m)
        println("Timing direct n = $n, m = $m...")
        direct_timings[i, j] = @belapsed begin
            for k in eachindex($ws)
                @inbounds begin
                    wk = $ws[k]
                    @simd for j in eachindex(rs)
                        $out_buffer[k] += $in_buffer[j] * exp(im*wk*$rs[j]) 
                    end
                end
            end
        end
    end
end

##

timing_ratios = direct_timings ./ nufft_timings

heatmap(
    timing_ratios, 
    ylabel="number of targets", yticks=(1:2:length(ns), string.(ns)[1:2:end]),
    xlabel="number of sources", xticks=(1:2:length(ms), string.(ms)[1:2:end]),
    clims=(0, 100)
    )