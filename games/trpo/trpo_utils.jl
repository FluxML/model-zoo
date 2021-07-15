"""
Utility functions specific to TRPO
"""

"""
Obtain the gradient vector product
"""

function gvp(policy,states,kl_vars,x)
    """
    Intermediate utility function, calculates Σ∇D_kl*x
    
    x : Variable to be estimated using conjugate gradient (Hx = g); (NUM_PARAMS,1)
    """
    model_params = get_policy_params(policy)
    gs = Tracker.gradient(() -> kl_loss(policy,states,kl_vars),model_params;nest=true)

    flat_grads = get_flat_grads(gs,get_policy_net(policy))
    return sum(x' * flat_grads)
end

function Hvp(policy,states,kl_vars,x;damping_coeff=0.1)
    """
    Computes the Hessian Vector Product
    Hessian is that of the kl divergence between the old and the new policies wrt the policy parameters
    
    Returns : Hx; H = ∇²D_kl
    """
    model_params = get_policy_params(policy)
    hessian = Tracker.gradient(() -> gvp(policy,states,kl_vars,x),model_params)
    return get_flat_grads(hessian,get_policy_net(policy)) # .+ (damping_coeff .* x)
end

function conjugate_gradients(policy,states,kl_vars,Hvp,b,nsteps=10,err=1e-10)
    """
    b : Array of shape (NUM_PARAMS,1)

    Solves x for Hx = b
    """
    
    x = zeros(size(b))
    
    r = copy(b)
    p = copy(b)
    
    rdotr = r' * r
    
    for i in 1:nsteps
        hvp = Hvp(policy,states,kl_vars,p).data # Returns array of shape (NUM_PARAMS,1)

        α = rdotr ./ (p' * hvp)
        
        x = x .+ (α .* p)
        r = r .- (α .* hvp)

        new_rdotr = r' * r
        β = new_rdotr ./ rdotr
        p = r .+ (β .* p)
        
        rdotr = new_rdotr
        
        if rdotr[1] < err
            break
        end
    end
    
    return x
end
