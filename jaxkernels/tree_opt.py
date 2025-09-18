import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
from jsindy.util import tree_scale,tree_dot,tree_add
from warnings import warn

def build_armijo_linesearch(f,decrease_ratio=0.5,slope=0.05,max_iter = 25):
    def armijo_linesearch(x, f_curr, d, g, t0=0.1):
        """
        x: current parameters (pytree)
        f_curr: f(x)
        d: descent direction (pytree)
        g: gradient at x (pytree)
        t0: initial step size
        a: Armijo constant
        """
        candidate = tree_add(x, tree_scale(d, -t0))
        dec0 = f(candidate) - f_curr
        pred_dec0 = -t0 * tree_dot(d, g)
        
        # The loop state: (iteration, t, current decrease, predicted decrease)
        init_state = (0, t0, dec0, pred_dec0)

        def cond_fun(state):
            i, t, dec, pred_dec = state
            # Continue while we haven't satisfied the Armijo condition and haven't exceeded max_iter iterations.
            not_enough_decrease = dec >= slope * pred_dec
            return jnp.logical_and(i < max_iter, not_enough_decrease)

        def body_fun(state):
            i, t, dec, pred_dec = state
            t_new = decrease_ratio*t
            candidate_new = tree_add(x, tree_scale(d, -t_new))
            dec_new = f(candidate_new) - f_curr
            pred_dec_new = -t_new * tree_dot(d, g)
            return (i + 1, t_new, dec_new, pred_dec_new)

        # Run the while loop
        i_final, t_final, dec_final, pred_dec_final = jax.lax.while_loop(
            cond_fun, body_fun, init_state
        )
        armijo_rat_final = dec_final/pred_dec_final
        candidate_final = tree_add(x, tree_scale(d, -t_final))
        return candidate_final, t_final,armijo_rat_final
    return armijo_linesearch


def run_gradient_descent(loss, init_params, init_stepsize=0.001, maxiter=10000, tol=1e-6,show_progress=True,**kwargs):
    params = init_params
    losses = []
    step_sizes = []
    gnorms = []

    loss_valgrad = jax.value_and_grad(loss)
    loss_fun = loss
    armijo_linesearch = build_armijo_linesearch(loss_fun,**kwargs)
    t = init_stepsize

    @jax.jit
    def gd_update(params,t):
        lossval, g = loss_valgrad(params)
        new_params,new_t,armijo_rat = armijo_linesearch(params,lossval,g,g,t0 = t)
        gnorm = jnp.sqrt(tree_dot(g,g))
        return new_params,new_t,gnorm,lossval,armijo_rat
    
    if show_progress:
        wrapper = tqdm
    else:
        wrapper = lambda x: x

    for i in wrapper(range(maxiter)):
        params,t,gnorm,lossval,armijo_rat = gd_update(params,t)
        if armijo_rat<0.01:
            warn("Line search failed")
        if i>0:
            if lossval>losses[-1]:
                print(lossval)
        losses.append(lossval)
        step_sizes.append(t)
        gnorms.append(gnorm)
        if gnorm<tol:
            break
        if armijo_rat>0.5:
            t = 1.2*t
        if armijo_rat<0.1:
            t = t/2


    conv_history = {
        'values':jnp.array(losses),
        'stepsizes':jnp.array(step_sizes),
        'gradnorms':jnp.array(gnorms)
    }
    return params,conv_history
        
def run_jaxopt_solver(solver, x0, show_progress=True):
    state = solver.init_state(x0)
    sol = x0
    values, errors, stepsizes = [state.value], [state.error], [state.stepsize]
    num_restarts = 0

    @jax.jit
    def update(sol, state):
        return solver.update(sol, state)

    if show_progress:
        wrapper=tqdm
    else:
        wrapper = lambda x: x

    for iter_num in wrapper(range(solver.maxiter)):
        sol, state = update(sol, state)
        values.append(state.value)
        errors.append(state.error)
        stepsizes.append(state.stepsize)
        if solver.verbose > 0:
            print("Gradient Norm: ", state.error)
            print("Loss Value: ", state.value)
        if state.error <= solver.tol:
            break
        if stepsizes[-1] == 0:
            num_restarts = num_restarts+1
            print(f"Restart {num_restarts}")
            if num_restarts>10:
                break
            state = solver.init_state(sol)
    convergence_data = {
        "values": jnp.array(values),
        "gradnorms": jnp.array(errors),
        "stepsizes": jnp.array(stepsizes),
    }
    return sol, convergence_data, state
