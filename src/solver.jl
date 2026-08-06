module Solver

export FAS!

using LinearAlgebra

"""
    ```
        function FAS(
            f, 
            Q::AbstractArray{Tf};
            coarseners = [], prolongators = [],
            multigrid_level::Int = 0,
            n_iter::Int = 50,
            rtol::Real = 1.0f-1, atol::Real = 1.0f-7,
        ) where {Tf <: AbstractFloat}
    ```

Run Full Approximation Scheme (FAS) over function `f` such that:

```
l = 1 # multigrid level

r, ω = f(l, Q)
Q .+= ω .* r # fixed-point iteration
```

Where `Q` is the state variable vector,
`f` is a function that returns its desired, fixed-point-iteration values,
and `ω` is a relaxation coefficient or array of relaxation coefficients.

Returns residual L2 norm reduction ratio.

If many multigrid levels are given, provide an array of coarseners and an
array of prolongators. They must be callables such that `coarseners[i](x)`
coarsens array `x` from level `i-1` to level `i`, and `prolongators[i](x)`
prolongates array `x` from level `i` to level `i-1`.
"""
function FAS!(
    f, Q::AbstractArray{Tf};
    coarseners = [], prolongators = [],
    perscribed_f::Union{AbstractArray, Nothing} = nothing,
    multigrid_level::Int = 0,
    n_iter::Int = 50,
    rtol::Real = 1.0f-1, atol::Real = 1.0f-7,
) where {Tf <: AbstractFloat}
    l = multigrid_level

    fQ, ω = f(l, Q)

    source = 0.0f0
    if !isnothing(perscribed_f)
        source = perscribed_f .- fQ
    end

    r = fQ .+ source
    nr0 = norm(r)
    nr = nr0

    if length(coarseners) > 1
        coars = coarseners[1]
        prolong = prolongators[1]

        Qc = coars(Q)
        Qcold = copy(Qc)

        pfQc = coars(r)

        FAS!(
            f, Qc;
            coarseners = coarseners[2:end], prolongators = prolongators[2:end],
            perscribed_f = pfQc, multigrid_level = multigrid_level + 1,
            n_iter = n_iter, atol = atol, rtol = rtol,
        )

        Q .+= prolong(Qc .- Qcold)
    end

    for _ = 1:n_iter
        r, ω = f(l, Q)
        @. r += source
        @. Q += clamp(ω, 0.0f0, 1.0f0) * r

        nr = norm(r)
        if nr < nr0 * rtol + atol
            break
        end
    end

    nr / (nr0 + eps(typeof(nr0)))
end

end
