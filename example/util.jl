using FastHankelTransform

function skeleton_plot(rs, ws; z_split=FastHankelTransform.NUFHT_Z_SPLIT[])
    return skeleton_plot(FastHankelTransform.generate_boxes(rs, ws, z_split=z_split))
end

function skeleton_plot(boxes::Tuple{Vector{NTuple{4, Int64}}, Vector{NTuple{4, Int64}}, Vector{NTuple{4, Int64}}})
    n = maximum(vcat(map(box_set -> getindex.(box_set, 2), boxes)...))
    pl = plot(xticks=([],[]), yticks=([],[]), axis=([], false))
    for (box_set, color) in zip(boxes, [:red, :dodgerblue, :black])
        for box in box_set
            i0b, i1b, j0b, j1b = box
            plot!(pl, 
                [j0b, j1b, j1b, j0b, j0b], 
                [n-i0b+1, n-i0b+1, n-i1b+1, n-i1b+1, n-i0b+1],
                c=color, line=(color == :black ? (2, :dash) : (2, :solid)), label="")
        end
    end

    return pl
end