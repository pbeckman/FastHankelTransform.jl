using FastHankelTransform, Test
using FastHankelTransform.LinearAlgebra

@testset "All tests" begin
    ws = collect(range(0, 189, 171))
    rs = collect(range(0, 153, 212))
    cs = collect(range(-563, 829, 212))

    @testset "Integer ν accuracy" begin
        for tol in 10.0 .^ (-4:-1:-14)
            for nu in FastHankelTransform.nus_int
                gs_true = FastHankelTransform.besselj.(nu, ws .* rs') * cs
                gs      = nufht(nu, rs, cs, ws; tol=tol)
                @test norm(gs_true - gs) / norm(gs_true) < 10*tol
            end
        end
    end
end
