struct ThreeLayerNetwork{F}

    Din::Int64
    H1::Int64
    H2::Int64
    Dout::Int64
    f::F
end

ThreeLayerNetwork(;in::Int64 = in, out::Int64 = out, H1::Int64 = H1, H2::Int64=H2, f = tanh) = ThreeLayerNetwork(in, H1, H2, out, f)

#----------------------------------------------------
function Base.show(io::IO, a::ThreeLayerNetwork)
#----------------------------------------------------
    print(io, "ThreeLayerNetwork with ",a.Din ," inputs, ", a.H1, " hidden neurons followed by ", a.H2," hidden neurons and ",a.Dout, " outputs.\nActivation function is ",a.f)
end


#----------------------------------------------------
function (a::ThreeLayerNetwork)(weights, x)
#----------------------------------------------------

    MARK = 0

    W1 = reshape(weights[MARK+1:MARK + a.H1 * a.Din], a.H1, a.Din)

    MARK += a.H1 * a.Din

    b1 = weights[MARK+1:MARK + a.H1]

    MARK += a.H1

    W2 = reshape(weights[MARK+1:MARK + a.H2 * a.H1], a.H2, a.H1)

    MARK += a.H2 * a.H1

    b2 = weights[MARK+1:MARK + a.H2]

    MARK += a.H2

    W3 = reshape(weights[MARK+1:MARK + a.Dout * a.H2], a.Dout, a.H2)

    MARK += a.Dout * a.H2

    b3 = weights[MARK+1:MARK + a.Dout]

    MARK += a.Dout

    # @assert(MARK == length(weights)) # make sure all weights have been used up

    # x is Din × N
    # W1*x .+ b is H × N

    W3 * a.f.(W2 * a.f.(W1*x .+ b1) .+ b2) .+ b3
   
end

numweights(a::ThreeLayerNetwork) = (a.H1 * a.Din + a.H1) + (a.H1 * a.H2 + a.H2) + (a.H2 * a.Dout + a.Dout)