module QuantumTimeEvolution

include("Operators.jl")    

export exactEvolve
export exactevolvebosehubbard
export krylovEvolve

function exactEvolve(Ψ, H, t, dt)
    M = exp(-im * dt * H)
    result = [deepcopy(Ψ)]
    for i in dt:dt:t
        Ψ .= M * Ψ
        push!(result, deepcopy(Ψ))
    end
    return result
end

function exactevolvebosehubbard(d, L, Ψ, dt, t; w = 1, U = 1, J = 1)
    H = bosehubbard(d, L; w = w, U = U, J = J)
    return exactEvolve(Ψ, H, t, dt)
end

function exactEvolveWithCheb(Ψ, H, t, dt)
    M = expChebApproximant(dt * H)
    result = [deepcopy(Ψ)]
    for i in dt:dt:t
        Ψ .= M * Ψ
        push!(result, deepcopy(Ψ))
    end
    return result
end

function seriesEvolve(Ψ, H, t, dt, k)
    result = [deepcopy(Ψ)]
    M = -im * dt * H
    for _ in dt:dt:t
        Ψtemp = result[end]
        Ψ .= Ψtemp
        for i in 1:k-1
            Ψtemp = M * Ψtemp
            Ψ .+= Ψtemp ./ factorial(i)
        end
        push!(result, deepcopy(Ψ))
    end
    return result
end

function krylovEvolve(Ψ, H, t, dt, k)
    result = [deepcopy(Ψ)]
    for i in dt:dt:t
        Hₖ, U = krylovSubspace(result[end], H, k)
        push!(result, normalize(U * exp(-1im * dt * Hₖ)[:, 1]))
    end
    return result
end

function krylovSubspace(Ψ, H, k)
    Hₖ = complex(zeros(k, k))
    U = complex(zeros(length(Ψ), k))

    U[:, 1] = normalize(Ψ)
    z = H * U[:, 1]
    for i in 1:k-1
        a = U[:, i]' * z
        z .-= a * U[:, i]
        b = norm(z)
        Hₖ[i, i] = a
        Hₖ[i, i + 1] = b
        Hₖ[i + 1, i] = b
        U[:, i + 1] .= z / b
        z = H * U[:, i + 1] .- b * U[:, i]
    end
    Hₖ[k, k] = U[:, k]' * z
    return Hₖ, U
end

function expChebApproximant(A) #A needs to be skew hermitian, ie. -iH, where H is hermitian
    methodValues = [2, 4, 8, 12, 18];
    θ = 100 .* [1.38e-05, 2.92e-02, 0.1295, 0.636, 2.212] #Error bounds corresponding to different approximants

    normA = norm(A, 1)#Matrix norm is used as a bound to the Error    
    if(normA <= θ[end])
        for method in 1:length(methodValues)
            if normA <= θ[method]
                return chebApproximants(A, methodValues[method])
            end
        end
    else
        chebApproximants(A, 18)
        #throw(DomainError(normA, "The matrix norm is too large, it is not possible to calculate the approximation withouth error. Consider decreasing dt."))
    end
end

function chebApproximants(A, order)
    if (order >= 2)
        A2 = A*A; 
    end
    if (order > 8)
        A3 = A*A2;
    end
    if(order == 2) 
        alp_0 = 0.999999999999999999999811107325001498863;
        alp_1 = -0.999999999976195000000188892674999250568*1im;
        alp_2 = -0.499999999992065000000047223168749850113;
        return alp_0*I + alp_1*A+alp_2*A2;
    elseif(order == 4) 
        x1 = 0.16666657785001893215842467829973122728*1im;
        x2 = 0.04166664890333648869312416038897725009;
        alp_0 = 0.99999999999999999997309614471715549922;
        alp_1 = -0.99999999999981067844714105106755532083*1im;
        alp_2 = -0.49999999999994320353145192968957453728;               
            
        A4 = A2*(x1*A + x2*A2);
        return alp_0*I + alp_1*A + alp_2*A2 + A4;
    elseif(order == 8)
        # P_8(A) with theta=0.1295
        x1 = 431/4000;    
        x2 = -0.02693906873598870733*1im;
        x3 = 0.66321004441662438593*1im;
        x4 = 0.54960853911436015786*1im;
        x5 = 0.16200952846773660904;
        x6 = -0.01417981805211804396*1im;
        x7 = -0.03415953916892111403;
        alp_0 = 0.99999999999999999928;
        alp_1 = -0.99999999999999233987*1im;
        alp_2 = -0.13549409636220703066;
        # Matrix products
        A4 = A2*(x1*A + x2*A2);
        A8 = (x3*A2 + A4)*(x4*I + x5*A + x6*A2 + x7*A4) ;
        return alp_0*I + alp_1*A + alp_2*A2 + A8;
    elseif(order == 12)
        # P_12(A) with theta=0.636
        a01 = -6.26756985350202252845;
        a11 = 2.52179694712098096140*1im;
        a21 = 0.05786296656487001838;
        a31 = -0.07766686408071870344*1im;
        a02 = 0;
        a12 = 1.41183797496250375498*1im;
        a22 = 0;
        a32 = -0.00866935318616372016*1im;
        a03 = 2.69584306915332564689; 
        a13 = -1.35910926168869260391*1im; 
        a23 = -0.09896214548845831754; 
        a33 = 0.01596479463299466666*1im; 
        a04 = 0;
        a14 = 0.13340427306445612526*1im; 
        a24 = 0.02022602029818310774; 
        a34 = -0.00674638241111650999*1im;
        
        B1 = a01*I+a11*A+a21*A2+a31*A3;
        B2 = a02*I+a12*A+a22*A2+a32*A3;
        B3 = a03*I+a13*A+a23*A2+a33*A3;
        B4 = a04*I+a14*A+a24*A2+a34*A3;
        # Matrix products
        A6 = B3 + B4*B4;
        return B1 + (B2 + A6)*A6;
    elseif(order == 18)
        # P_18(A) with theta=2.212
        a01 = 0; 
        a11 = 3/25;
        a21 = -0.00877476096879703859*1im;
        a31 = -0.00097848453523780954;
        b01 = 0;
        b11 = -0.66040840760771318751*1im;
        b21 = -1.09302278471564897987;
        b31 = 0.25377155817710873323*1im;
        b61 = 0.00054374267434731225;
        b02 = -2.58175430371188142440;
        b12 = -1.73033278310812419209*1im;
        b22 = -0.07673476833423340755;
        b32 = -0.00261502969893897079*1im;
        b62 = -0.00003400011993049304;
        b03 = 2.92377758396553673559;
        b13 = 1.44513300347488268510*1im;
        b23 = 0.12408183566550450221;
        b33 = -0.01957157093642723948*1im;
        b63 = 0.00002425253007433925;
        b04 = 0;
        b14 = 0;
        b24 = -0.123953695858283131480*1im;
        b34 = -0.011202694841085592373;
        b64 = -0.000012367240538259896*1im;

        # Matrix products
        A6 = A3*A3;
        B1 = (a01*I + a11*A .+ a21*A2 .+ a31*A3);
        B2 = (b01*I + b11*A .+ b21*A2 .+ b31*A3 .+ b61*A6); 
        B3 = (b02*I + b12*A .+ b22*A2 .+ b32*A3 .+ b62*A6);
        B4 = (b03*I + b13*A .+ b23*A2 .+ b33*A3 .+ b63*A6);
        B5 = (b04*I + b14*A .+ b24*A2 .+ b34*A3 .+ b64*A6);
        
        A9 = B1*B5 .+ B4;
        return B2 .+ (B3 .+ A9)*A9;
    end
end

end # module
