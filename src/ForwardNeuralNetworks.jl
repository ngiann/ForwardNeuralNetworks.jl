module ForwardNeuralNetworks

    import Base: Base.==

    include("neuralnetwork.jl")

    export layer, getparam, setparam!, setparam, ==, numweights

end
