struct example{F}
    
    f::F

end

function (e::example)(x)

    e.f(x)

end

function (V::Vector{example})(x)

    out = x

    for v in V

        out = v(out)

    end

    out

end

function chainedevaluation(T,x)
    out = x
    for t in T
        out = t(out)
    end
    out
end

