# For recreating ComponentArrays with a different copy_with_eltype
function copy_with_eltype(x, T)
    x_new = similar(x, T)
    x_new .= x
    return x_new
end