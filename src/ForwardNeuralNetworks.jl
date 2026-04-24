module ForwardNeuralNetworks

    using Random
    using LinearAlgebra
    # using Optim

    # include("neuralnetwork.jl")
    # include("LinearNetwork.jl")
    # include("TwoLayerNetwork.jl") 
    # include("ThreeLayerNetwork.jl")  

    include("newcode.jl")



    export TwoLayerNetwork, ThreeLayerNetwork
    export BufferedTwoLayerNetwork, BufferedThreeLayerNetwork
    export forward!, numweights, inputdim, outputdim

end
