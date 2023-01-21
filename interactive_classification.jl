### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# ╔═╡ 96e5d714-9937-11ed-3055-29b86c38a3a2
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()

    using Flux
    using Flux: onecold, onehotbatch, logitcrossentropy
	using WGLMakie
	import Flux3D
    using GraphNeuralNetworks
    using MLDatasets
    using MLUtils
    using LinearAlgebra, Random, Statistics
end

# ╔═╡ 79853901-2118-4bdf-85ca-25390f07ee26
begin
	batchsize = 16
	lr = 3e-4
	epochs = 100
	knn_K = 10
	num_classes = 10
	num_hidden = 32
	npoints = 512
	D = 500
	V = 500
end

# ╔═╡ 0a13d65c-d9fc-4515-a906-acf213e42a76
begin
	dset = Flux3D.ModelNet10(;
		download=true, 
		transform = Flux3D.TriMeshToPointCloud(npoints)
	)
	val_dset = Flux3D.ModelNet10(;
		train=false, 
		download=true, 
		transform = Flux3D.TriMeshToPointCloud(npoints)
	)
end

# ╔═╡ 5f0d7353-070e-496b-80ab-e1dcf50b5c70
pc = dset[23]

# ╔═╡ 0f463fe8-f79e-43d7-ad66-54e021bf0b0c
Flux3D.visualize(pc; markersize=1)

# ╔═╡ 4ec7403a-f04d-470e-a989-50a564c64897
begin
	train_inds = randperm(length(dset))[1:D]
	trainX = cat([dset[i].data.points for i in train_inds]..., dims=3)
	trainY = onehotbatch([dset[i].ground_truth for i in train_inds], 1:num_classes)

	val_inds = randperm(length(val_dset))[1:V]
	valX = cat([val_dset[i].data.points for i in val_inds]..., dims=3)
	valY = onehotbatch([val_dset[i].ground_truth for i in val_inds], 1:num_classes)
end

# ╔═╡ e428abb4-2b4c-49ff-88aa-897976901f60
begin
	train_dl = DataLoader((trainX, trainY); batchsize=batchsize, shuffle=true)
	val_dl = DataLoader((valX, valY); batchsize=batchsize)
end

# ╔═╡ a773c69a-a93e-44c8-8e4d-50128a78e75e
function to_knn_graph(x::AbstractArray{T,3}, K) where {T}
	batchsize = size(x, 3)
	gi = zeros(Int, size(x)[2:3]) .+ reshape(1:batchsize, 1, batchsize)
	gi = reshape(gi, :)
	x = reshape(x, size(x, 1), :)
	return knn_graph(x, K; graph_indicator=gi), x
end

# ╔═╡ 8c954064-baf0-4e00-9805-bc46408b6a7d
begin 
	x, y = first(train_dl)
	g, x = to_knn_graph(x, knn_K)
end

# ╔═╡ 16fe84e0-3d98-409b-9c06-223b6de11d23
l = EdgeConv(Dense(6=>8), +)

# ╔═╡ 878b8dc5-202f-4af4-b3bc-f7ded4bc04c1
x′ = l(g, x)

# ╔═╡ 00281f4f-624d-476d-b3d6-6a44784da6e4
model = GNNChain(
	EdgeConv(Dense(6=>num_hidden), +), 
	BatchNorm(num_hidden),
	x -> relu.(x),
	EdgeConv(Dense(2num_hidden=>num_hidden), +),
	BatchNorm(num_hidden),
	x -> relu.(x), 
	EdgeConv(Dense(2num_hidden=>num_hidden), +),
	BatchNorm(num_hidden), 
	GlobalPool(mean),
	Dropout(0.5),
	Dense(num_hidden=>num_classes)
)

# ╔═╡ 77f099f1-3dde-4941-a9a8-665799dfe894
accuracy(yhat, y) = mean(onecold(yhat, 1:num_classes) .== onecold(y, 1:num_classes))

# ╔═╡ f34f00d0-55fb-4a68-8651-9db4d632e213
st_opt = Flux.setup(Adam(lr), model)

# ╔═╡ eaf61d28-a436-465e-8265-c6a8b3bd2340
out = model(g, x)

# ╔═╡ d140ea70-b947-4f5e-b702-efc8567c3e7b
for epoch=1:epochs
	total_loss = 0
	total_acc = 0
	for (x, y) in train_dl
		g, x = to_knn_graph(x, knn_K)
		loss, ∇ = Flux.withgradient(model) do m
			yhat = m(g, x)
			logitcrossentropy(yhat, y)
		end
		total_loss += loss
		Flux.update!(st_opt, model, ∇[1])
	end

	for (x, y) in val_dl
		g, x = to_knn_graph(x, knn_K)
		yhat = model(g, x)
		total_acc += accuracy(yhat, y)
	end
	total_acc /= length(val_dl)
	@info "$epoch: loss=$total_loss, val_acc=$total_acc"
end

# ╔═╡ 391b3adf-bc96-4e61-8efc-043b39ba7494
begin
	xt, yt = first(val_dl)
	gt, xt = to_knn_graph(xt, knn_K)
	yhat = model(g, xt)
	acc = accuracy(yhat, yt)
end

# ╔═╡ ab17bdb2-8c89-4a6d-8771-c285809a2ce3
gval, xval = to_knn_graph(valX, knn_K)

# ╔═╡ 16894712-163a-4711-a794-f68223bd1f89
yhatval = model(gval, xval)

# ╔═╡ 42570b79-59ad-4b21-88c8-93d96583fc80
yhatc = onecold(yhatval)

# ╔═╡ Cell order:
# ╠═96e5d714-9937-11ed-3055-29b86c38a3a2
# ╠═79853901-2118-4bdf-85ca-25390f07ee26
# ╠═0a13d65c-d9fc-4515-a906-acf213e42a76
# ╠═5f0d7353-070e-496b-80ab-e1dcf50b5c70
# ╠═0f463fe8-f79e-43d7-ad66-54e021bf0b0c
# ╠═4ec7403a-f04d-470e-a989-50a564c64897
# ╠═e428abb4-2b4c-49ff-88aa-897976901f60
# ╠═a773c69a-a93e-44c8-8e4d-50128a78e75e
# ╠═8c954064-baf0-4e00-9805-bc46408b6a7d
# ╠═16fe84e0-3d98-409b-9c06-223b6de11d23
# ╠═878b8dc5-202f-4af4-b3bc-f7ded4bc04c1
# ╠═00281f4f-624d-476d-b3d6-6a44784da6e4
# ╠═77f099f1-3dde-4941-a9a8-665799dfe894
# ╠═f34f00d0-55fb-4a68-8651-9db4d632e213
# ╠═eaf61d28-a436-465e-8265-c6a8b3bd2340
# ╠═d140ea70-b947-4f5e-b702-efc8567c3e7b
# ╠═391b3adf-bc96-4e61-8efc-043b39ba7494
# ╠═ab17bdb2-8c89-4a6d-8771-c285809a2ce3
# ╠═16894712-163a-4711-a794-f68223bd1f89
# ╠═42570b79-59ad-4b21-88c8-93d96583fc80
