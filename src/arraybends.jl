module ArrayBackends

    using DocStringExtensions

    using Base: Meta

    export to_backend, @declare_converter

    """
    $TYPEDSIGNATURES

    Convert array to given to backend
    """
    to_backend(a::AbstractArray, converter) = converter(a)

    """
    $TYPEDSIGNATURES

    Convert tuple to given to backend
    """
    to_backend(a::Tuple, converter) = map(
        v -> to_backend(v, converter), a
    )

    """
    $TYPEDSIGNATURES

    Convert dictionary to given to backend
    """
    to_backend(a::AbstractDict, converter) = Dict(
        [
            k => to_backend(v, converter) for (k, v) in pairs(a)
        ]...
    )

    """
    $TYPEDSIGNATURES

    Convert any other type to backend (do nothing)
    """
    to_backend(a::Any, converter) = a

    """
    $TYPEDSIGNATURES

    Overload `to_backend` to struct type `T`.
    Declares method which runs backend conversion to all properties
    and rebuilds the struct
    """
    macro declare_converter(T)
        T = esc(T)
        qualified_name = esc(GlobalRef(ArrayBackends, :to_backend))

        quote
            """
            $TYPEDSIGNATURES

            Method for backend conversion of custom struct type
            """
            function $qualified_name(s::$T, converter)
                args = []
                for f in fieldnames($T)
                    push!(
                        args, $qualified_name(getfield(s, f), converter)
                    )
                end
                $T(args...)
            end
        end 
    end 

end