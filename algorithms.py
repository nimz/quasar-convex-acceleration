import numpy as np
import torch


SQRT_FLT_EPS = torch.tensor(np.sqrt(np.finfo(np.float32).eps))
SQRT_DBL_EPS = torch.tensor(np.sqrt(np.finfo(np.float64).eps), dtype=torch.double)

class AGD_options:
    def __init__(self):
        self.max_iters = 2000
        self.tol = 1e-4
        self.tol_type = 'grad'
        self.lr = 1
        self.step_type = 'adaptive'
        self.step_inc = 1.1
        self.step_dec = 0.6
        self.max_ss_iters = 100
        self.max_as_iters = 100
        self.reduced_tau = True
        self.restart = 'none'
        self.num_tol = 1e-8
        self.mode = 'standard'
        self.ls_mode = 'exact'
        self.ls_guess = False
        self.guess_first = False
        self.verbose = False

def ssq(v):
    return torch.sum(v*v)

def incr(alpha, h):
    temp = alpha + h
    h = temp - alpha
    return temp, h

def binary_search(fg, x, v, L, b, c, fx, eps, max_iters, reduced_tau,
                  grad_mode, guess=None, guess_first=False):
    fd = grad_mode != 'exact'
    num_eps = SQRT_FLT_EPS if x.dtype == torch.float else SQRT_DBL_EPS
    def g(alpha, func_only=False, fval=None):
        assert (0 <= alpha <= 1)
        w = alpha*x + (1-alpha)*v
        if func_only:
            return fg(w, func_only=True)
        if not fd:
            if fval is not None:
                G_f = fg(w, grad_only=True)
            else:
                fval, G_f = fg(w)
            dg = torch.dot(G_f, x-v)
        else:
            fval = fval if fval is not None else fg(w, func_only=True)
            alpha2, h = incr(alpha, num_eps*alpha)
            w2 = alpha2*x + (1-alpha2)*v
            f2val = fg(w2, func_only=True)
            dg = (f2val - fval) / h
        return fval, dg, None if fd else G_f
    xv_sqdist = ssq(x-v)
    if xv_sqdist < num_eps**2 or torch.norm((x-v)/x, float('inf')) < num_eps:
        return 1, fx, None  # avoid line search if x, v very close
    p = b*xv_sqdist
    if guess_first and guess is not None and guess != 1:
        g_1 = fx
        g_guess, dg_guess, G_guess = g(guess)
        if c*g_guess + guess*(dg_guess - guess*p) <= c*g_1 + eps:
            return guess, g_guess, G_guess
    g_1, dg_1, G_1 = g(1, fval=fx)
    if dg_1 <= eps + p:
        return 1, g_1, G_1
    g_0 = g(0, func_only=True)
    if c == 0 or g_0 <= g_1 + eps/c:
        return 0, g_0, None
    if not guess_first and guess is not None:
        g_guess, dg_guess, G_guess = g(guess)
        if c*g_guess + guess*(dg_guess - guess*p) <= c*g_1 + eps:
            return guess, g_guess, G_guess
    if reduced_tau:
        tau = 1 - (eps+p) / (L*xv_sqdist)
        g_tau, dg_tau, G_tau = g(tau)
    else:
        tau = 1
        g_tau, dg_tau, G_tau = g_1, dg_1, G_1
    tau = torch.tensor(tau, dtype=x.dtype)
    alpha, g_alpha, dg_alpha, G_alpha = tau, g_tau, dg_tau, G_tau
    lo, hi = torch.tensor(0, dtype=x.dtype), tau
    n_iters = 0
    while c*g_alpha + alpha*(dg_alpha - alpha*p) > c*g_1 + eps and n_iters < max_iters:
        alpha = (lo + hi) / 2
        g_alpha, dg_alpha, G_alpha = g(alpha)
        if g_alpha <= g_tau:
            hi = alpha
        else:
            lo = alpha
        n_iters += 1
    return alpha.item(), g_alpha, G_alpha

def agd_framework(fg, x0, beta, etafun, cfun, alphafun, ls_tol, options):
    if options.step_type == 'search':
        raise NotImplementedError('Full step size search not implemented!')
    K = options.max_iters
    step_size = options.lr
    evals = [0, 0]
    fg_old = fg
    def fg(x, func_only=False, grad_only=False):
        evals[0] += not grad_only
        evals[1] += not func_only
        if grad_only:
            return fg_old(x)[1]
        return fg_old(x, func_only=func_only)

    L = 1 / step_size
    eta = etafun(L, 0)
    x, y, v = x0, x0, x0
    f_y, df_y = 0, 0
    f_x_new = fg(x, func_only=True)
    fvals = [f_x_new]

    take_step = True
    num_nostep = 0
    k = 0
    while k < K:
        if options.verbose: print(f'Step {k}')
        if take_step:
            g_alpha, G_alpha = None, None
            if options.mode == 'standard':
                if beta == 0:
                    alpha = 1
                else:
                    b = (1-beta)/(2*eta)
                    c = cfun(k)
                    alpha, g_alpha, G_alpha = binary_search(fg, x, v, L, b, c, f_x_new, ls_tol, options.max_as_iters,
                            options.reduced_tau, options.ls_mode, guess=alphafun(k) if options.ls_guess else None,
                            guess_first=options.guess_first)
            elif options.mode == 'convex':
                alpha = alphafun(k)
            elif options.mode == 'agmsdr':
                raise NotImplementedError('AGMSDR not implemented!')
            if alpha == 1 or (options.mode == 'convex' and torch.norm(x-v) == 0):
                # latter check is for consistency of our algorithm and regular AGD
                y = x
                f_y = f_x_new
                df_y = fg(y, grad_only=True)
            else:
                y = alpha*x + (1-alpha)*v
                if options.mode == 'standard':
                    f_y = g_alpha if g_alpha is not None else fg(y, func_only=True)
                    df_y = G_alpha if G_alpha is not None else fg(y, grad_only=True)
                else:
                    f_y, df_y = fg(y)
            if options.tol and options.tol_type == 'grad' and torch.max(torch.abs(df_y)) < options.tol:
                fvals.append(f_y)
                k += 1
                break

        theta = step_size
        L = 1/theta
        eta = etafun(L, k)

        x_new = y - theta*df_y
        v_new = beta*v + (1-beta)*y - eta*df_y

        take_step = True
        f_x_new = fg(x_new, func_only=True)
        if options.verbose: print(f'Loss: {f_x_new}')
        if options.tol and options.tol_type == 'func' and f_x_new < options.tol:
            fvals.append(f_x_new)
            k += 1
            break
        delta = (f_y - theta*ssq(df_y)/2) - f_x_new
        if delta >= 0:
            if options.step_type != 'constant':
                step_size *= options.step_inc
        else:
            if delta < -options.num_tol and options.step_type == 'constant':
                raise ValueError('Constant step size too large')
            step_size *= options.step_dec
            take_step = False
            num_nostep += 1

        do_restart = (options.restart == 'alpha' and alpha == 1) or \
                     (options.restart == 'grad' and torch.dot(df_y, x-v) / torch.norm(df_y) < 0) or \
                     (options.restart == 'fval' and f_y > f_x)
        take_step = take_step and not do_restart

        if take_step:
            x, v = x_new, v_new
            fvals.append(f_x_new)
            k += 1
    func_eval, grad_eval = evals
    return k, num_nostep, func_eval, grad_eval, fvals, x

def agd_strong(fg, gamma, L, mu, x0, options=AGD_options()):
    options.step_type, options.lr = 'constant', 1/L
    beta = 1 - gamma*np.sqrt(mu/L)
    ls_tol = 0
    def etafun(L, k):
        return np.sqrt(1 / (mu*L))
    kappa = np.sqrt(L/mu)
    def cfun(k):
        return kappa
    def alphafun(k):
        return kappa / (1+kappa)
    return agd_framework(fg, x0, beta, etafun, cfun, alphafun, ls_tol, options)

def agd_nonstrong(fg, gamma, x0, options=AGD_options()):
    options.reduced_tau = False
    beta = 1
    ls_tol = gamma*options.tol/2
    omegas = [1]
    def omegafun(k):
        assert (len(omegas)-2 <= k <= len(omegas)-1)
        if k == len(omegas) - 1:
            omega = omegas[-1]
            omega = omega/2*(np.sqrt(omega**2+4) - omega)
            omegas.append(omega)
        return omegas[k+1]
    def etafun(L, k):
        omega = omegafun(k)
        return gamma / (L*omega)
    def cfun(k):
        omega = omegafun(k)
        return gamma*(1/omega-1)
    def alphafun(k):
        omega = omegafun(k)
        return 1-omega
    return agd_framework(fg, x0, beta, etafun, cfun, alphafun, ls_tol, options)

def gd(fg, x0, options=AGD_options()):
    K = options.max_iters
    step_size = options.lr
    evals = [0, 0]
    fg_old = fg
    def fg(x, func_only=False, grad_only=False):
        evals[0] += not grad_only
        evals[1] += not func_only
        if grad_only:
            return fg_old(x)[1]
        return fg_old(x, func_only=func_only)

    L = 1 / step_size
    x = x0
    f_x_new = fg(x, func_only=True)
    fvals = [f_x_new]

    take_step = True
    num_nostep = 0
    k = 0

    while k < K:
        if options.verbose: print(f'Step {k}')
        if take_step:
            f_x = f_x_new
            df_x = fg(x, grad_only=True)
            if options.tol and (options.tol_type == 'grad' and torch.max(torch.abs(df_x)) < options.tol) \
                or (options.tol_type == 'func' and f_x < options.tol):
                fvals.append(f_x)
                k += 1
                break

        theta = step_size
        x_new = x - theta*df_x

        take_step = True
        f_x_new = fg(x_new, func_only=True)
        if options.verbose: print(f'Loss: {f_x_new}')
        delta = (f_x - theta*ssq(df_x)/2) - f_x_new
        if delta >= 0:
            if options.step_type != 'constant':
                step_size *= options.step_inc
        else:
            if delta < -options.num_tol and options.step_type == 'constant':
                raise ValueError('Constant step size too large')
            step_size *= options.step_dec
            take_step = False
            num_nostep += 1

        if take_step:
            x = x_new
            fvals.append(f_x_new)
            k += 1

    func_eval, grad_eval = evals
    return k, num_nostep, func_eval, grad_eval, fvals, x
