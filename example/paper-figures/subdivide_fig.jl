using FastHankelTransform, FastGaussQuadrature, SpecialFunctions, Plots, Plots.Measures, LaTeXStrings

function number_boxes(n, m, boxes; w=0)
    M = fill(NaN, n, m)

    for (box_set, val) in zip(boxes, [1,2,3])
        for (b, box) in enumerate(box_set)
            i0b, i1b, j0b, j1b   = box
            M[i0b:i1b, j0b:j1b] .= val
            M[i0b:i1b, max(1,j0b-w):min(m,j0b+w)] .= 0
            M[i0b:i1b, max(1,j1b-w):min(m,j1b+w)] .= 0
            M[max(1,i0b-w):min(n,i0b+w), j0b:j1b] .= 0
            M[max(1,i1b-w):min(n,i1b+w), j0b:j1b] .= 0
        end
    end

    return M
end

# order of transform
nu  = 0
# number of sources
m   = 1_000
# number of targets
n   = 1_000
# tolerance
tol = 1e-15

FastHankelTransform.setup_nufht!(nu, tol)
# don't allow small local boxes to be labeled as direct
FastHankelTransform.NUFHT_LOC_K[] = 0

# get crossover point
z = FastHankelTransform.NUFHT_Z_SPLIT[]

# choose some points and frequencies
rs = 10 .^ collect(range(0.3, 2, m))
ws = collect(range(0, 20, n))

# compute splitting indices for every row and column 
js_split = [findfirst(>(z / w), rs) for w in ws]
is_split = (1:n)[(!).(isnothing.(js_split))]
js_split = js_split[(!).(isnothing.(js_split))]

# set up plotting
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)
gr(size=(300, 300))

p0 = heatmap(
    besselj.(nu, ws*rs'), 
    yflip=true, 
    axis=([], false),
    c=reverse(cgrad(:Spectral)),
    margin=0mm,
    colorbar=false
    )
savefig("./figures/A.pdf")

# width of box boundaries
w = 4

# plot true matrix splitting
p1 = heatmap(
    (1 .+ (ws*rs' .< z)) .* ([zeros(2w); ones(m-4w); zeros(2w)]) .* ([zeros(2w); ones(m-4w); zeros(2w)])', 
    yflip=true, 
    legend=:none, axis=([], false),
    c=[:black, scrungle[6], scrungle[7]],
    margin=0mm
    )
plot!(p1, js_split, is_split, c=:black, line=4, widen=false)
savefig("./figures/splitting.pdf")

# plot box subdivision at various levels
num_levels = 4
pls = Vector{Plots.Plot}(undef, num_levels)
for level=1:num_levels
    boxes = FastHankelTransform.generate_boxes(
    rs, ws, 
    z_split=z, min_dim_prod=100,
    max_levels=level
    )
    local pl = heatmap(
        number_boxes(n, m, boxes, w=4) .* ([zeros(2w); ones(m-4w); zeros(2w)]) .* ([zeros(2w); ones(m-4w); zeros(2w)])', 
        yflip=true, 
        legend=:none, axis=([], false),
        c=[:black, scrungle[7], scrungle[6], :gray25]
        )
    plot!(pl, js_split, is_split, c=:black, line=4, widen=false)
    pls[level] = pl
    savefig("./figures/splitting_lvl$level.pdf")
end