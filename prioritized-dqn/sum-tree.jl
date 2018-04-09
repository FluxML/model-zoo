mutable struct SumTree
    capacity::Int
    tree::Array{Float64, 1}
    data::Array{Any, 2}
    data_pointer::Int

    function SumTree(cap::Int, state_size::Int)
        capacity = cap
        tree = zeros(2 * capacity - 1)
        data = zeros(2 * state_size + 2, capacity)
        data_pointer = 0
        new(capacity, tree, data, data_pointer)
    end
end

function add!(tree::SumTree, p, data)
    tree_idx = tree.data_pointer + tree.capacity - 1
    tree.data[tree.data_pointer + 1] = data
    update!(tree, tree_idx, p)  # update tree_frame

    tree.data_pointer += 1
    if tree.data_pointer >= tree.capacity  # replace when exceed the capacity
        tree.data_pointer = 0
    end
end

function update!(tree::SumTree, tree_idx::Int, p)
    change = p - tree.tree[tree_idx + 1]
    tree.tree[tree_idx + 1] = p
    # then propagate the change through tree
    while tree_idx != 0    # this method is faster than the recursive loop in the reference code
        tree_idx = div(tree_idx, 2)
        tree.tree[tree_idx + 1] += change
    end
end

function get_leaf(tree::SumTree, v)
    parent_idx = 0
    leaf_idx = 0
    while true
        cl_idx = 2 * parent_idx + 1        # this leaf's left and right kids
        cr_idx = cl_idx + 1
        if cl_idx >= length(tree.tree)        # reach bottom, end search
            leaf_idx = parent_idx
            break
        elseif v <= tree.tree[cl_idx + 1]
            parent_idx = cl_idx
        else
            v -= tree.tree[cl_idx + 1]
            parent_idx = cr_idx
        end
    end

    data_idx = leaf_idx - tree.capacity + 1
    return leaf_idx, tree.tree[leaf_idx + 1], tree.data[data_idx + 1]
end

function total_p(tree::SumTree)
    return tree.tree[1]  # the root
end
