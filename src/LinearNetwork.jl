struct LinearNetwork

    Din::Int64
    Dout::Int64

end

LinearNetwork(;in::Int64 = in, out::Int64 = out) = LinearNetwork(in, out)

#----------------------------------------------------
function Base.show(io::IO, a::LinearNetwork)
#----------------------------------------------------
    print(io, "LinearNetwork with ",a.Din ," inputs and ",a.Dout, " outputs.")
end


#----------------------------------------------------
function (a::LinearNetwork)(weights, x)
#----------------------------------------------------

    MARK = 0

    W1 = @views reshape(weights[MARK+1:MARK + a.Dout * a.Din], a.Dout, a.Din)

    MARK += a.Dout * a.Din

    b1 = @views weights[MARK+1:MARK + a.Dout]

    MARK += a.Dout

    W1*x .+ b1
   
end

numweights(a::LinearNetwork) = a.Dout * a.Din + a.Dout