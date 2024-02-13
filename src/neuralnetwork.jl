#########################################################
# Constructor
#########################################################

struct layer{F}
    Din::Int64  # number of inputs to layer
    Dout::Int64 # number of outputs of layer
    f::F
end


function makelayer(;out=nou::Int64, in=nin::Int64, f = tanh)
    
    layer(in, out, f)

end


##########################################################
# REPL
##########################################################

function Base.show(io::IO, a::layer)
    print(io, "layer with ",a.Din ," inputs and ",a.Dout, " outputs.\nFunction is ",a.f)
end


#########################################################
# Evaluation
#########################################################

#----------------------------------------------------
function (a::layer)(weights, x)
#----------------------------------------------------

    MARK = 0

    aux1 = @view weights[MARK+1:MARK +  a.Dout * a.Din]

    w = reshape(aux1, a.Dout, a.Din)

    MARK += a.Dout * a.Din

    aux2 = @view weights[MARK+1:MARK + a.Dout]
    
    b = reshape(aux2, a.Dout)

    MARK += a.Dout

    @assert(MARK == length(weights))

    a.f.(w * x .+ b) # evaluation of layer

end


#----------------------------------------------------
function (a::Array{layer,1})(weights, x)
#----------------------------------------------------

    MARK = 0

    out = x

    for l in a
        
        # the output of each layer is the input of the next layer
        # the input to the first layer is x
        
        out = @views l(weights[MARK+1:MARK+numweights(l)], out)
        
        MARK += numweights(l)

    end

    @assert(MARK == numweights(a)) # make sure all weights have been used up

    return out

end




#########################################################
# Number of parameters in network
#########################################################

numweights(a::layer)          = a.Din * a.Dout + a.Dout

numweights(a::Array{layer,1}) = mapreduce(numweights, +, a)

numlayers(a::Array{layer,1})  = length(a)