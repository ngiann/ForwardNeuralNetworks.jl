module ForwardNeuralNetworks

    using Random
    using LinearAlgebra
    # using Optim

    # include("neuralnetwork.jl")
    include("TwoLayerNetwork.jl") 
    include("ThreeLayerNetwork.jl")  

    export numweights, TwoLayerNetwork, ThreeLayerNetwork

end
