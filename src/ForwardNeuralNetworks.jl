module ForwardNeuralNetworks

    using Random
    using LinearAlgebra
    # using Optim

    # include("neuralnetwork.jl")
    include("LinearNetwork.jl")
    include("TwoLayerNetwork.jl") 
    include("ThreeLayerNetwork.jl")  

    export numweights, LinearNetwork, TwoLayerNetwork, ThreeLayerNetwork

end
