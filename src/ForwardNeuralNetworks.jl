module ForwardNeuralNetworks

    using Random
    using LinearAlgebra
    using Optim

    include("neuralnetwork.jl")
    include("test.jl")

    export numweights, test, makelayer

end
