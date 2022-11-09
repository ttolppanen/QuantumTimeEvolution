using LinearAlgebra

export ⊗, numberOperator, hamiltonian

⊗(a, b) = kron(a, b)

function annihilationOperator(d)
    m = zeros(d, d)
    for i in 1:d-1
        m[i, i+1] = sqrt(i)
    end
    return m
end

function numberOperator(d)
    a = annihilationOperator(d)
    return a' * a
end

function hamiltonian(d, s)::Matrix{Complex{Float64}}
    a = annihilationOperator(d)
    n = numberOperator(d)
    H = zeros(d^s, d^s)
    for i in 1:s
        ñ = Matrix(I, d^(i-1), d^(i-1)) ⊗ n ⊗ Matrix(I, d^(s-i), d^(s-i))
        H .+= ñ
        if(i != s)
            hopping = Matrix(I, d^(i-1), d^(i-1)) ⊗ a ⊗ a' ⊗ Matrix(I, d^(s-i - 1), d^(s-i - 1))
            H .+= hopping + hopping'
        end
    end
    return H
end

function bosehubbard(d, L; w=1, U=1, J=1)
    a = annihilationOperator(d)
    n = numberOperator(d)
    H = zeros(d^L, d^L)
    for i in 1:L
        ñ = Matrix(I, d^(i-1), d^(i-1)) ⊗ n ⊗ Matrix(I, d^(L-i), d^(L-i))
        H .+= w * ñ
        H .+= -U/2 * ñ * (ñ - I)
        if(i != L)
            hopping = Matrix(I, d^(i-1), d^(i-1)) ⊗ a ⊗ a' ⊗ Matrix(I, d^(L-i - 1), d^(L-i - 1))
            H .+= J * (hopping + hopping')
        end
    end
    return H
end