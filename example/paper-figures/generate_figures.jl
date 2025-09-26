using Plots, Printf

if !isdir("./figures")
    mkdir("./figures")
end

# define color palette used in figures
scrungle = parse.(RGB, [
    "#114982" # blue
    "#FF8080" # red
    "#00A61C" # green
    "#E2BFFF" # purple
    "#CCBC43" # yellow
    "#58BADB" # cyan
    "#B80056" # wine
    "#BDE861" # lime
    "#B05B00" # orange
    "#8B00DB" # magenta
    "#DBDBDB" # grey
])

@printf("\n\nGENERATING FIGURE 1\n\n")
include("two_expansions_fig.jl")

@printf("\n\nGENERATING FIGURE 2\n\n")
include("subdivide_fig.jl")

@printf("\n\nGENERATING FIGURES 3, 4, & 5 (this will take minutes to hours)\n\n")
include("performance_fig.jl")

@printf("\n\nGENERATING FIGURE 6 (this will take minutes to hours)\n\n")
include("fourier_scaling_fig.jl")

@printf("\n\nGENERATING FIGURE 7 (this will take a few minutes)\n\n")
include("fourier_bessel_fig.jl")