import matplotlib.pyplot as plt
import numpy as np
import itertools

import jax
import jax.numpy as jnp

def gradFirstOrder(apply_fn):
    fn = apply_fn
    fn = jax.jacrev(fn, argnums=1)
    fn = jax.vmap(fn, in_axes=(None,0))
    return fn

def gradSecondOrder(apply_fn):
    fn = apply_fn
    fn = jax.jacrev(fn, argnums=1)
    fn = jax.jacrev(fn, argnums=1)
    fn = jax.vmap(fn, in_axes=(None,0))
    return fn

def gradThirdOrder(apply_fn):
    fn = apply_fn
    fn = jax.jacrev(fn, argnums=1)
    fn = jax.jacrev(fn, argnums=1)
    fn = jax.jacrev(fn, argnums=1)
    fn = jax.vmap(fn, in_axes=(None,0))
    return fn

def heat_eq_1d(apply_fn, params, x, alpha=1/100):
    alpha = alpha
    du_dt = gradFirstOrder(apply_fn)(params, x)[:,0,1]
    du_dxx = gradSecondOrder(apply_fn)(params, x)[:,0,0,0]
    return du_dt-alpha*du_dxx

def laplace_2d(apply_fn, params, x):
    du_dxx = gradSecondOrder(apply_fn)(params, x)[:,0,0,0]
    du_dyy = gradSecondOrder(apply_fn)(params, x)[:,0,1,1]
    return du_dxx + du_dyy

class Condition():
    def __init__(self, fn, dims = (), bool_cond_fn = None):
        self.buffer = None
        self.dims = dims
        self.apply_fn = fn
        self.where = bool_cond_fn

    def __call__(self, x, buffer = False):
        if self.where:
            X = x[self.where(x)]
        else:
            X = x
        res = (X,self.apply_fn(X))
        if buffer: self.buffer = res

        return res

from functools import partial

class CauchyProblem():
    def __init__(self, domain, res_fun, bcs=[], ics=[], indomain_conditions=[], measurements = None):
        self.domain = domain
        self.res_fun = res_fun
        self.time = domain.time_dependent
        self.dim = domain.dim
        self.data = measurements

        self.bcs = bcs
        self.ics = ics
        self.indomain_conditions = indomain_conditions

        self.is_ic = True
        self.is_bc = True
        self.is_conds = False

        if len(bcs)==0:
            self.is_bc = False
        if len(ics)==0:
            self.is_ic = False
        if len(indomain_conditions)>0:
            self.is_conds = True

        if self.time:
            self.space_dim = domain.geometry.dim
            self.geo = domain.geometry
        else:
            self.space_dim = domain.dim
            self.geo = domain

    @partial(jax.jit, static_argnames=['self', 'apply_fn'])
    def residual(self, apply_fn, params, x):
        return self.res_fun(apply_fn, params, x)

    def res_mse(self, apply_fn, params, n_res):
        return self.res_loss(apply_fn, params, n_res, mse_loss)
    
    def res_loss(self, apply_fn, params, n_res, loss_fn):
        if self.time:
            x = self.domain.get_sample(n_res, 0, 0)['domain']
        else:
            x = self.domain.get_sample(n_res, 0)['domain']

        loss = loss_fn(self.residual(apply_fn, params, x), jnp.zeros_like(x[:,0:1]))
        return loss

    def res_kare(self, apply_fn, params, n_res):
        if self.time:
            x = self.domain.get_sample(n_res, 0, 0)['domain']
        else:
            x = self.domain.get_sample(n_res, 0)['domain']

        def pinn_fn(params, x):
            return self.residual(apply_fn, params, x).reshape((-1,1))

        loss = self._kare_loss(pinn_fn, params, x, jnp.zeros_like(x[:,0]))
        return loss

    def buffer_bc(self):
        x = self.domain.points_basin['boundary']

        for f in self.bcs:
            f(x, buffer = True)

    def buffer_ic(self):
        x = self.domain.points_basin['ic']

        for f in self.ics:
            f(x, buffer = True)

    def buffer_indomain(self):
        x = self.domain.points_basin['domain']

        for f in self.indomain_conditions:
            f(x, buffer = True)

    def data_mse(self, apply_fn, params):
        x = self.data[0]
        y = self.data[1]
        loss = mse_loss(apply_fn(params, x), y)
        return loss

    def _kare_loss(self, apply_fn, params, x, y):
        K = compute_ntk(apply_fn, params, x, x)
        loss = kare(apply_fn, y.squeeze(), K, 0.01)
        return loss

    def _conditions_kare(self, apply_fn, params, Ns=[], conds=[], sample_dom='domain', from_buff=False):
        if not len(Ns)==len(conds):
            raise ValueError("Number of N's values must equal that of conditions list !")
        if not loss_fn:
            loss_fn = mse_loss

        loss = 0

        if from_buff:
            l = len(conds[0].buffer[0])
            if l<1:
                raise ValueError("No buffer prepared for a condition !")
            idx = np.random.choice(l, Ns[0])
            X, Y = (conds[0].buffer[0][idx], conds[0].buffer[1][idx])
            dims = conds[0].dims
            loss = self._kare_loss(apply_fn, params, X, Y[:,dims])

            for f, n in zip(conds[1:], Ns[1:]):
                l = len(f.buffer[0])
                if l<1:
                    raise ValueError("No buffer prepared for a condition !")
                idx = np.random.choice(l, n)
                X, Y = (f.buffer[0][idx], f.buffer[1][idx])
                dims = f.dims
                loss += self._kare_loss(apply_fn, params, X, Y[:,dims])
        else:
            x = self.domain.get_sample(n, 0, 0)[sample_dom]

            X, Y = conds[0](x)
            dims = conds[0].dims
            loss = self._kare_loss(apply_fn, params, X, Y[:,dims])

            for f in conds[1:]:
                X, Y = f(x)
                dims = f.dims
                loss += self._kare_loss(apply_fn, params, X, Y[:,dims])

        return loss

    def _conditions_loss(self, apply_fn, params, Ns=[], conds=[], loss_fn=None, sample_dom='domain', from_buff=False):
        if not len(Ns)==len(conds):
            raise ValueError("Number of N's values must equal that of conditions list !")
        if not loss_fn:
            loss_fn = mse_loss

        loss = 0

        if from_buff:
            l = len(conds[0].buffer[0])
            if l<1:
                raise ValueError("No buffer prepared for a condition !")
            idx = np.random.choice(l, Ns[0])
            X, Y = (conds[0].buffer[0][idx], conds[0].buffer[1][idx])
            dims = conds[0].dims
            loss = loss_fn(apply_fn(params, X)[:,dims], Y[:,dims])

            for f, n in zip(conds[1:], Ns[1:]):
                l = len(f.buffer[0])
                if l<1:
                    raise ValueError("No buffer prepared for a condition !")
                idx = np.random.choice(l, n)
                X, Y = (f.buffer[0][idx], f.buffer[1][idx])
                dims = f.dims
                loss += loss_fn(apply_fn(params, X)[:,dims], Y[:,dims])
        else:
            x = self.domain.get_sample(n, 0, 0)[sample_dom]

            X, Y = conds[0](x)
            dims = conds[0].dims
            loss = loss_fn(apply_fn(params, X)[:,dims], Y[:,dims])

            for f in conds[1:]:
                X, Y = f(x)
                dims = f.dims
                loss += loss_fn(apply_fn(params, X)[:,dims], Y[:,dims])

        return loss

    def _conditions_mse(self, apply_fn, params, Ns=[], conds=[], sample_dom='domain', from_buff=False):
        return self._conditions_loss(apply_fn, params, Ns, conds, mse_loss, sample_dom, from_buff)

    def indomain_conds_mse(self, apply_fn, params, Ns, from_buff=False):
        if not self.is_conds:
            raise TypeError("No additional conditions have been specified !")

        return self._conditions_mse(apply_fn, params, Ns, conds=self.conds, sample_dom='domain', from_buff=from_buff)

    def indomain_conds_loss(self, apply_fn, params, Ns, loss_fn, from_buff=False):
        if not self.is_conds:
            raise TypeError("No additional conditions have been specified !")

        return self._conditions_loss(apply_fn, params, Ns, conds=self.conds, loss_fn=loss_fn, sample_dom='domain', from_buff=from_buff)

    def ic_mse(self, apply_fn, params, Ns, from_buff=False):
        if not self.time:
            raise ValueError("The problem must be time-dependent !")

        if not self.is_ic:
            raise TypeError("No Ic's have been specified !")

        return self._conditions_mse(apply_fn, params, Ns, conds=self.ics, sample_dom='ic', from_buff=from_buff)

    def ic_loss(self, apply_fn, params, Ns, loss_fn, from_buff=False):
        if not self.time:
            raise ValueError("The problem must be time-dependent !")

        if not self.is_ic:
            raise TypeError("No Ic's have been specified !")

        return self._conditions_loss(apply_fn, params, Ns, conds=self.ics, loss_fn=loss_fn, sample_dom='ic', from_buff=from_buff)

    def bc_mse(self, apply_fn, params, Ns, from_buff=False):
        if not self.is_bc:
            raise TypeError("No boundary conditions have been specified !")

        return self._conditions_mse(apply_fn, params, Ns, conds=self.bcs, sample_dom='boundary', from_buff=from_buff)

    def bc_loss(self, apply_fn, params, Ns, loss_fn, from_buff=False):
        if not self.is_bc:
            raise TypeError("No boundary conditions have been specified !")

        return self._conditions_loss(apply_fn, params, Ns, conds=self.bcs, loss_fn=loss_fn, sample_dom='boundary', from_buff=from_buff)

@jax.jit
def mse_loss(x, y):
    return jnp.mean((x-y)**2)

from functools import partial

@partial(jax.jit, static_argnums=(0,))
def compute_gradient(apply_fn, params, xs) -> jnp.ndarray:
    grad_fn = jax.grad(lambda p, x: apply_fn(p, x).squeeze())

    def _pointwise(x):
        flat_grads = []
        for leaf in jax.tree_leaves(grad_fn(params, x)):
            flat_grads.append(leaf.flatten())
        return jnp.concatenate(flat_grads)
    
    per_sample = jax.vmap(_pointwise, in_axes=0)
    return per_sample(xs)

@partial(jax.jit, static_argnums=(0,))
def compute_ntk(
    apply_fn,
    params,
    xs1: jnp.ndarray,
    xs2: jnp.ndarray,
) -> jnp.ndarray:
    
    G1 = compute_gradient(apply_fn, params, xs1)
    G2 = compute_gradient(apply_fn, params, xs2)
    return G1.dot(G2.T)

@partial(jax.jit, static_argnums=(0,))
def kare(apply_fn, y: jnp.ndarray, K: jnp.ndarray, z: float) -> float:
    n = K.shape[0]
    K_norm = K / n
    mat = K_norm + z * jnp.eye(n)
    inv = jax.jit(jnp.linalg.inv, backend="cpu")(mat)
    inv2 = inv @ inv
    return (((1/n) * y @ inv2 @ y.T) / ((1/n) * jnp.trace(inv)) ** 2)