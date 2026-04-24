module ForwardNeuralNetworks

  
    using LinearAlgebra
  
    include("newcode.jl")
    include("showmethods.jl")


    export TwoLayerNetwork, ThreeLayerNetwork
    export BufferedTwoLayerNetwork, BufferedThreeLayerNetwork
    export forward!, numweights, inputdim, outputdim, workspace, basenet, remake

end
