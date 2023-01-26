module DGCNN

function pointer(graphx, Φx, graphy, Φy, X, Y)
    @assert graphx.num_graphs == graphy.num_graphs

    batchsize = graphx.num_graphs
    for b=1:batchsize
        gix = graphx.graph_indicator .== b
        giy = graphy.graph_indicator .== b
        x   = X[:, gix]  # 3 x N1
        y   = Y[:, giy]  # 3 x N2
        ϕx = Φx[:, gix]  # f x N1
        ϕy = Φy[:, giy]  # f x N2
        M = softmax(ϕx' * ϕy, dims=1)  # N1 x N2
        yhat = M * y'

        # SVD
        x̄ = mean(x, dims=2)
        ȳ = mean(y, dims=2)
        H = (x .- x̄)' * (y .- ȳ)  # N1 x N2
        U, _, V = svd(H)
        R = V*U'        # 3 x 3
        t = -R*x̄ .+ ȳ   # 3 x 1
    end
    
    return cat(R_vector..., dims=3), cat(t_vector..., dims=3)
end

end # module DGCNN
