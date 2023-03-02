import torch
from torch.autograd.functional import vjp

from .fixed_grid import Euler
from .misc import _check_inputs, _flat_to_shape

SOLVERS = {
    'euler': Euler
}


def odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`, in either increasing or decreasing order. The first element of
            this sequence is taken to be the initial time point.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
        event_fn: Function that maps the state `y` to a Tensor. The solve terminates when
            event_fn evaluates to zero. If this is not None, all but the first elements of
            `t` are ignored.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
    """

    shapes, func, y0, t, rtol, atol, method, options, t_is_reversed = _check_inputs(func, y0, t, rtol, atol, method, options, SOLVERS)

    solver = SOLVERS['euler'](func=func, y0=y0, rtol=rtol, atol=atol, **options)

    solution, solution_dt = solver.integrate(t)


    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    return solution, solution_dt



class ImplicitFnGradientRerouting(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, event_fn, event_t, state_t):
        """ event_t is the solution to event_fn """
        ctx.func = func
        ctx.event_fn = event_fn
        ctx.save_for_backward(event_t, state_t)
        return event_t.detach(), state_t.detach()

    @staticmethod
    def backward(ctx, grad_t, grad_state):
        func = ctx.func
        event_fn = ctx.event_fn
        event_t, state_t = ctx.saved_tensors

        event_t = event_t.detach().clone().requires_grad_(True)
        state_t = state_t.detach().clone().requires_grad_(True)

        f_val = func(event_t, state_t)

        with torch.enable_grad():
            c, (par_dt, dstate) = vjp(event_fn, (event_t, state_t))

        # Total derivative of event_fn wrt t evaluated at event_t.
        dcdt = par_dt + torch.sum(dstate * f_val)

        # Add the gradient from final state to final time value as if a regular odeint was called.
        grad_t = grad_t + torch.sum(grad_state * f_val)

        dstate = dstate * (-grad_t / (dcdt + 1e-12)).reshape_as(c)

        grad_state = grad_state + dstate

        return None, None, None, grad_state
