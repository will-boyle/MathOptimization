import numpy as np

'''
One-hidden-layer ReLU neural network trained by gradient descent.

=============================================================================
ARCHITECTURE
=============================================================================

    y_hat(x) = sum_j  v_j * ReLU(w_j * x + b_j),   j = 1 ... K

Each neuron j applies a ReLU activation to its linear input:

    a_j(x) = ReLU(w_j * x + b_j)   where  ReLU(z) = max(0, z)

The network output is a weighted sum of these piecewise-linear responses.
With enough neurons it can approximate any continuous function on a bounded
domain (universal approximation theorem).

=============================================================================
LOSS AND GRADIENTS
=============================================================================

The training loss is mean squared error over n data points:

    L(W, b, v) = (1/n) * sum_i ( y_hat(x_i) - y_i )^2

Let r_i = y_hat(x_i) - y_i  (residual at point i).  The gradient with
respect to each parameter follows from the chain rule:

    dL/dv_j  =  (2/n) * sum_i  r_i * a_j(x_i)

    dL/dw_j  =  (2/n) * sum_i  r_i * v_j * ReLU'(z_j(x_i)) * x_i

    dL/db_j  =  (2/n) * sum_i  r_i * v_j * ReLU'(z_j(x_i))

where  z_j(x) = w_j * x + b_j,  a_j(x) = ReLU(z_j(x)),
and  ReLU'(z) = 1 if z > 0 else 0.

=============================================================================
INITIALIZATION
=============================================================================

W ~ N(0, 0.5): random slopes.

b = -W * U, U ~ Uniform(0, 1): each neuron's breakpoint (-b/w) is placed
uniformly at random inside the normalized training domain [0, 1].  A neuron
with breakpoint outside [0, 1] is either always-on or always-off, contributing
no gradient signal — this "dead neuron" problem wastes capacity.

v ~ N(0, 0.5 / sqrt(K)): output weights are scaled down by sqrt(K) so the
initial network output has variance ~0.25 regardless of K.  Without this,
a large K causes a large initial loss and gradient explosion on step 1.

=============================================================================
ALGORITHM
=============================================================================

Vanilla gradient descent:

    W <- W - alpha * dL/dW
    b <- b - alpha * dL/db
    v <- v - alpha * dL/dv

Repeated for n_steps iterations.
'''


def _relu(z):
    return np.maximum(0.0, z)


def _relu_d(z):
    return (z > 0).astype(float)


def predict_nn(x, W, b, v, x_shift=0.0, x_scale=1.0, y_shift=0.0, y_scale=1.0):
    """Network prediction at a single scalar input x (handles normalization)."""
    x_norm = (x - x_shift) / x_scale
    y_norm = float(v @ _relu(W * x_norm + b))
    return y_norm * y_scale + y_shift


def _mse_norm(W, b, v, x_norm, y_norm):
    return float(np.mean(
        [(float(v @ _relu(W * x + b)) - y) ** 2
         for x, y in zip(x_norm, y_norm)]
    ))


def _gradients_norm(W, b, v, x_norm, y_norm):
    n  = len(x_norm)
    gW = np.zeros_like(W)
    gb = np.zeros_like(b)
    gv = np.zeros_like(v)
    for x, y in zip(x_norm, y_norm):
        z     = W * x + b
        a     = _relu(z)
        r     = float(v @ a) - y
        dL    = 2.0 * r / n
        gv   += dL * a
        chain = dL * v * _relu_d(z)
        gW   += chain * x
        gb   += chain
    return gW, gb, gv


def train(x_data, y_data, n_neurons=50, alpha=0.1, n_steps=10000,
          seed=0, verbose=True):
    '''
    Train a one-hidden-layer ReLU network by gradient descent.

    Parameters
    ----------
    x_data    : array-like (n,)  input values
    y_data    : array-like (n,)  target values
    n_neurons : int              number of hidden neurons  (default 50)
    alpha     : float            learning rate             (default 0.1)
    n_steps   : int              gradient descent steps    (default 10000)
    seed      : int              random seed               (default 0)
    verbose   : bool             print convergence table   (default True)

    Returns
    -------
    W, b, v        : learned weights (operate on normalized inputs)
    x_shift, x_scale, y_shift, y_scale : normalization constants
    losses         : list of (step, loss) pairs at logged checkpoints

    Use predict_nn(x, W, b, v, x_shift, x_scale, y_shift, y_scale) to
    evaluate the trained network on new inputs.
    '''
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    K      = n_neurons

    # Normalize inputs and outputs to [0, 1]
    x_shift, x_scale = x_data.min(), x_data.max() - x_data.min()
    y_shift, y_scale = y_data.min(), y_data.max() - y_data.min()
    if x_scale == 0: x_scale = 1.0
    if y_scale == 0: y_scale = 1.0
    x_norm = (x_data - x_shift) / x_scale
    y_norm = (y_data - y_shift) / y_scale

    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.5, K)
    U = rng.uniform(0.0, 1.0, K)   # breakpoints drawn uniformly from training domain
    b = -W * U
    v = rng.normal(0.0, 0.5 / np.sqrt(K), K)

    # Checkpoints to log
    log_at = {0, 1, 2, 3, 4, 5}
    for s in [10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000]:
        if s <= n_steps:
            log_at.add(s)
    log_at.add(n_steps)

    losses = []

    if verbose:
        print()
        print(f"    {K} neurons  |  alpha = {alpha}  |  {n_steps} steps  |  n = {len(x_data)} points")
        print()
        print(f"    {'Step':>6}  {'Loss':>16}")
        print(f"    {'------':>6}  {'----------------':>16}")

    for step in range(n_steps + 1):
        L = _mse_norm(W, b, v, x_norm, y_norm)
        if step in log_at:
            losses.append((step, L))
            if verbose:
                print(f"    {step:>6}  {L:>18.8f}")
        if step < n_steps:
            gW, gb, gv = _gradients_norm(W, b, v, x_norm, y_norm)
            W -= alpha * gW
            b -= alpha * gb
            v -= alpha * gv

    return W, b, v, x_shift, x_scale, y_shift, y_scale, losses
