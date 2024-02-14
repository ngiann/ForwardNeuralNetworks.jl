struct TwoLayerNetwork{F}

    Din::Int64
    H::Int64
    Dout::Int64
    f::F
end

TwoLayerNetwork(;in::Int64 = in, out::Int64 = out, H::Int64 = H, f = tanh) = TwoLayerNetwork(in, H, out, f)

#----------------------------------------------------
function Base.show(io::IO, a::TwoLayerNetwork)
#----------------------------------------------------
    print(io, "TwoLayerNetwork with ",a.Din ," inputs, ", a.H, " hidden neurons and ",a.Dout, " outputs.\nActivation function is ",a.f)
end


#----------------------------------------------------
function (a::TwoLayerNetwork)(weights, x)
#----------------------------------------------------

    MARK = 0

    W1 = reshape(weights[MARK+1:MARK + a.H * a.Din], a.H, a.Din)

    MARK += a.H * a.Din

    b1 = weights[MARK+1:MARK + a.H]

    MARK += a.H

    W2 = reshape(weights[MARK+1:MARK + a.Dout * a.H], a.Dout, a.H)

    MARK += a.Dout * a.H

    b2 = weights[MARK+1:MARK + a.Dout]

    MARK += a.Dout

    # @assert(MARK == length(weights)) # make sure all weights have been used up

    # x is Din × N
    # W1*x .+ b is H × N

    W2 * a.f.(W1*x .+ b1) .+ b2
   
end

numweights(a::TwoLayerNetwork) = a.H * a.Din + a.H + a.H * a.Dout + a.Dout