using Flux, GraphNeuralNetworks
import Flux3D
using WGLMakie
using Flux: onehotbatch, onecold, onehot, crossentropy, logitcrossentropy
using Statistics: mean
using Base.Iterators: partition

#Makie.inline!(false)
#Makie.set_theme!(show_axis = false)

batch_size = 1
lr = 3e-4
epochs = 5
K = 10
num_classes = 10
num_hidden = 32
npoints = 64
D = 100
V = 100

dset = Flux3D.ModelNet10(;download=true, transform = Flux3D.TriMeshToPointCloud(npoints))
val_dset = Flux3D.ModelNet10(;train=false, download=true, transform = Flux3D.TriMeshToPointCloud(npoints))
Flux3D.visualize(dset[11
valY = onehotbatch([val_dset[i].ground_truth for i=1:V], 1:num_classes)

TRAIN = [(cat(data[i]..., dims=3), labels[:, i]) for i in partition(1:D,  batch_size)]
VAL = (valX, valY)


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
        Dense(num_hidden, num_classes)
)

loss(yhat, y) = logitcrossentropy(yhat, y)
accuracy(yhat, y) = mean(onecold(yhat, 1:num_classes) .== onecold(y, 1:num_classes))

st_opt = Flux.setup(Adam(lr), model)

for epoch = 1:epochs
    running_loss = 0
    for (x, y) in TRAIN
        x = x[:,:,1]
        y = y[:,:,1]

        #@show size(x), size(y)

        g = knn_graph(x, 10)
        train_loss, ∇ = Flux.withgradient(model) do m
            yhat = m(g, x)
            loss(yhat, y)
        end

        running_loss += train_loss
        Flux.update!(st_opt, model, ∇[1])
    end
    println("Epoch: $(epoch), epoch_loss: $(running_loss)")#, accuracy: $(accuracy(model(valX), valY))\n")
end
#@show accuracy(model(valX), valY)

