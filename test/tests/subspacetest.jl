# using LinearAlgebra

@testset "subspace" begin

@testset "FindSubspace" begin
    d = 2
    L = 2
    find_subspace = generate_total_boson_number_subspace_finder(d, L)
    @test find_subspace(normalize!([1, 0, 0, 0])) == [1]
    @test find_subspace(normalize!([0, 1, 0, 0])) == [2, 3]
    @test find_subspace(normalize!([0, 0, 1, 0])) == [2, 3]
    @test find_subspace(normalize!([0, 0, 0, 1])) == [4]
    @test find_subspace(normalize!([0, 1.0, 1.0, 0])) == [2, 3]

    @test_throws ErrorException find_subspace(normalize!([1.0, 0, 0, 1.0])) == [2, 3]
end

end # testset