#########################################################
# Constructor
#########################################################

mutable struct layer
    Din::Int64
    Dout::Int64
    f::Function
end


function makelayer(;out=nou::Int64, in=nin::Int64, f = x -> tanh.(x) )
    
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

    a.f(w * x .+ b)

end


#----------------------------------------------------
function (a::Array{layer,1})(weights, x)
#----------------------------------------------------

    MARK = numweights(a)

    out  = a[end](weights[MARK-numweights(a[end])+1:MARK], x)

    MARK = MARK - numweights(a[end])

    for l in a[end-1:-1:1]
        out = l(weights[MARK-numweights(l)+1:MARK], out)
        MARK = MARK - numweights(l)
    end

    @assert(MARK == 0)
    return out

end




#########################################################
# Number of parameters in network
#########################################################

numweights(a::layer)          = a.Din * a.Dout + a.Dout

numweights(a::Array{layer,1}) = mapreduce(numweights, +, a)

numlayers(a::Array{layer,1})  = length(a)