import abc
import numpy as np
import itertools

import jax
import jax.numpy as jnp

import abc

def array32(x):
    return jnp.array(x, dtype=jnp.float32)


class Geometry(abc.ABC):
    """This is the base class for modular geometries, like the spacial part of it. Time dimension is not supported here. Know youre ABC's"""
    def __init__(self, dim, bbox, diam):
        self.dim = dim
        self.bbox = bbox
        self.diam = min(diam, np.linalg.norm(bbox[1] - bbox[0]))
        self.points_basin = {'domain': [], 'boundary': []}
        self.idstr = type(self).__name__
        self.time_dependent = False

    @property
    def basin_size(self):
        return {'domain': len(self.points_basin['domain']), 'boundary': len(self.points_basin['boundary'])}

    def get_sample(self, n_dmn, n_bnd, replace = True, verbose=False):
        if not replace and n_dmn>self.basin_size['domain']:
            raise ValueError("n_dmn={} samples exceed the basin size!".format(n_dmn))
        if not replace and n_bnd>self.basin_size['boundary']:
            raise ValueError("n_bnd={} samples exceed the basin size!".format(n_bnd))

        idx_dmn = np.random.choice(self.basin_size['domain'], n_dmn, replace=replace)
        idx_bnd = np.random.choice(self.basin_size['boundary'], n_bnd, replace=replace)

        if verbose and (n_dmn>=self.basin_size['domain'] or n_bnd>=self.basin_size['boundary']):
            print("Warning: The number of sample points is bigger than the basin!")

        return {'domain': np.array(self.points_basin['domain'])[idx_dmn], 'boundary': np.array(self.points_basin['boundary'])[idx_bnd]}

    def generate_basin(self, n_dmn, n_bnd, random="pseudo", verbose=False):
        self.points_basin['domain'] = self.random_points(n_dmn, random=random)
        self.points_basin['boundary'] = self.random_boundary_points(n_bnd, random=random)
        if verbose:
            print("Successfully generated basin points !")
        return self.points_basin

    @abc.abstractmethod
    def inside(self, x):
        """Is this point inside of the domain?"""

    @abc.abstractmethod
    def on_boundary(self, x):
        """Does this point lie on the boundary?"""

    def uniform_points(self, n, boundary=True):
        """Good for visualization maybe?"""
        print(
            "Warning: {}.uniform_points not implemented. Use random_points instead.".format(
                self.idstr
            )
        )
        return self.random_points(n)

    @abc.abstractmethod
    def random_points(self, n, random="pseudo"):
        """randomly sample points from the domain, so what will be residual loss points"""

    def uniform_boundary_points(self, n):
        """Compute the equispaced point locations on the boundary."""
        print(
            "Warning: {}.uniform_boundary_points not implemented. Used random_boundary_points instead.".format(
                self.idstr
            )
        )
        return self.random_boundary_points(n)

    @abc.abstractmethod
    def random_boundary_points(self, n, random="pseudo"):
        """Samle points on the boundary at random"""

    def union(self, other):
        return CSGUnion(self, other)

    def __or__(self, other):
        return CSGUnion(self, other)

    def difference(self, other):
        return CSGDifference(self, other)

    def __sub__(self, other):
        return CSGDifference(self, other)

    def intersection(self, other):
        return CSGIntersection(self, other)

    def __and__(self, other):
        return CSGIntersection(self, other)

class CSGUnion(Geometry):
    """Here we construct an object by CSG (Constructive Solid Geometry) union !"""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                "{} | {} failed (dimensions do not match).".format(
                    geom1.idstr, geom2.idstr
                )
            )
        super(CSGUnion, self).__init__(
            geom1.dim,
            (
                np.minimum(geom1.bbox[0], geom2.bbox[0]),
                np.maximum(geom1.bbox[1], geom2.bbox[1]),
            ),
            geom1.diam + geom2.diam,
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_or(self.geom1.inside(x), self.geom2.inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x)),
            np.logical_and(self.geom2.on_boundary(x), ~self.geom1.inside(x)),
        )

    def random_points(self, n, random="pseudo"):
        x = [np.zeros(shape=(n, self.dim))]
        i = 0
        while i < n:
            tmp1 = self.geom1.random_points(n//2, random=random)
            tmp2 = self.geom2.random_points(n//2, random=random)
            tmp = np.concatenate([tmp1, tmp2], axis=0)

            tmp1 = tmp[~CSGIntersection(self.geom1, self.geom2).inside(tmp)]
            tmp2 = tmp[CSGIntersection(self.geom1, self.geom2).inside(tmp)][::2]

            tmp = np.concatenate([tmp1, tmp2], axis=0)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return np.array(x)

    def random_boundary_points(self, n, random="pseudo"):
        x = np.zeros(shape=(n, self.dim))
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                ~self.geom1.inside(geom2_boundary_points)
            ]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return np.array(x)

class CSGDifference(Geometry):
    """Construct an object by CSG Difference."""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                "{} - {} failed (dimensions do not match).".format(
                    geom1.idstr, geom2.idstr
                )
            )
        super(CSGDifference, self).__init__(geom1.dim, geom1.bbox, geom1.diam)
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_and(self.geom1.inside(x), ~self.geom2.inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x)),
            np.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x)),
        )

    def random_points(self, n, random="pseudo"):
        x = np.zeros(shape=(n, self.dim))
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[~self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return np.array(x)

    def random_boundary_points(self, n, random="pseudo"):
        x = np.zeros(shape=(n, self.dim))
        i = 0
        while i < n:

            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.inside(geom2_boundary_points)
            ]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return np.array(x)


class CSGIntersection(Geometry):
    """Construct an object by CSG Intersection."""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                "{} & {} failed (dimensions do not match).".format(
                    geom1.idstr, geom2.idstr
                )
            )
        super(CSGIntersection, self).__init__(
            geom1.dim,
            (
                np.maximum(geom1.bbox[0], geom2.bbox[0]),
                np.minimum(geom1.bbox[1], geom2.bbox[1]),
            ),
            min(geom1.diam, geom2.diam),
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_and(self.geom1.inside(x), self.geom2.inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x)),
            np.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x)),
        )

    def random_points(self, n, random="pseudo"):
        x = np.zeros(shape=(n, self.dim))
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return np.array(x)

    def random_boundary_points(self, n, random="pseudo"):
        x = np.zeros(shape=(n, self.dim))
        i = 0
        while i < n:

            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.inside(geom2_boundary_points)
            ]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return np.array(x)

class Interval(Geometry):
    def __init__(self, l, r):
        '''Here, {l} is the left end of the interval and {r}, obviously, is the right!'''
        super(Interval, self).__init__(1, (np.array([l]), np.array([r])), r - l)
        self.l, self.r = l, r

    def inside(self, x):
        return np.logical_and(self.l <= x, x <= self.r).flatten()

    def on_boundary(self, x):
        return np.any(np.isclose(x, [self.l, self.r]), axis=-1)

    def uniform_points(self, n, boundary=True):
        if boundary:
            return np.linspace(self.l, self.r, num=n, )[:, None]
        return np.linspace(
            self.l, self.r, num=n + 1, endpoint=False,
        )[1:, None]

    def random_points(self, n, random="pseudo"):
        x = sample(n, 1, random)
        return (self.diam * np.array(x) + self.l)

    def uniform_boundary_points(self, n):
        if n == 1:
            return np.array([[self.l]])
        xl = np.full((n // 2, 1), self.l)
        xr = np.full((n - n // 2, 1), self.r)
        return np.vstack((xl, xr))

    def random_boundary_points(self, n, random="pseudo"):
        return np.random.choice([self.l, self.r], n)[:, None]

def sample(n_samples, d, sampler="pseudo"):
    if sampler == "pseudo":
        return pseudo(n_samples, d)
    raise ValueError("f{sampler} sampler is not available.")

def pseudo(n_samples, d):
    return np.random.random(size=(n_samples, d))


class TimeDomain(Interval):
    def __init__(self, t0, t1):
        super(TimeDomain, self).__init__(t0, t1)
        self.t0 = t0
        self.t1 = t1
        self.time_dependent = True

    def on_initial(self, t):
        return np.isclose(t, self.t0).flatten()


class GeometryXTime():
    def __init__(self, geometry, t0, t1):
        self.geometry = geometry
        self.timedomain = TimeDomain(t0, t1)
        self.dim = geometry.dim + self.timedomain.dim
        self.time_dependent = True
        self.points_basin = {'domain': [], 'boundary': [], 'ic': []}

    @property
    def basin_size(self):
        return {'domain': len(self.points_basin['domain']), 'boundary': len(self.points_basin['boundary']), 'ic': len(self.points_basin['ic'])}

    def generate_basin(self, n, random="pseudo", verbose=False):
        self.geometry.generate_basin(n, n, random=random, verbose=verbose)
        self.timedomain.generate_basin(n, 0, random=random, verbose=verbose)
        self.points_basin['domain'] = np.concatenate([self.geometry.points_basin['domain'], 
                                                      self.timedomain.points_basin['domain']], axis=1)
        self.points_basin['boundary'] = np.concatenate([self.geometry.points_basin['boundary'], 
                                                      self.timedomain.points_basin['domain']], axis=1)
        self.points_basin['ic'] = np.concatenate([self.geometry.points_basin['domain'], 
                                                      jnp.zeros((self.geometry.basin_size['domain'],1))], axis=1)
       
        if verbose:
            print("Successfully generated basin points !")

    def get_sample(self, n_dmn, n_bnd, n_ic, replace = True, verbose=False):
        smpl_dmn = np.concatenate([self.geometry.get_sample(n_dmn, 0, replace = replace, verbose=verbose)['domain'],
                                   self.timedomain.get_sample(n_dmn, 0, replace = replace, verbose=verbose)['domain']], axis=1)
        smpl_bnd = np.concatenate([self.geometry.get_sample(0, n_bnd, replace = replace, verbose=verbose)['boundary'],
                                   self.timedomain.get_sample(n_bnd, 0, replace = replace, verbose=verbose)['domain']], axis=1)
        smpl_ic = np.concatenate([self.geometry.get_sample(n_ic, 0, replace = replace, verbose=verbose)['domain'],
                                   np.zeros((n_ic,1))], axis=1)

        if verbose and (n_dmn>=self.basin_size['domain'] or n_bnd>=self.basin_size['boundary'] or n_ic>=self.basin_size['ic']):
            print("Warning: The number of sample points is bigger than the basin!")

        return {'domain': smpl_dmn, 'boundary': smpl_bnd, 'ic' : smpl_ic}


    def on_boundary(self, x):
        return self.geometry.on_boundary(x[:, :-1])

    def on_initial(self, x):
        return self.timedomain.on_initial(x[:, -1:])

    def uniform_points(self, n, boundary=True):
        """Uniform points on the spatio-temporal domain.

        Geometry volume ~ bbox.
        Time volume ~ diam.
        """
        nx = int((n * np.prod(self.geometry.bbox[1] - self.geometry.bbox[0]) / self.timedomain.diam)** 0.5)
        nt = n // nx
        x = self.geometry.uniform_points(nx, boundary=boundary)
        nx = len(x)
        if boundary:
            t = self.timedomain.uniform_points(nt, boundary=True)
        else:
            t = np.linspace(self.timedomain.t1, self.timedomain.t0, num=nt,endpoint=False)[:, None]
        xt = []
        for ti in t:
            xt.append(np.hstack((x, np.full([nx, 1], ti[0]))))
        xt = np.vstack(xt)
        if n != len(xt):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(xt))
            )
        return xt

    def random_points(self, n, random="pseudo"):
        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        t = np.random.default_rng().permutation(t)
        return np.hstack((x, t))

    def _cube_surface_area(bbox):
        '''returns surface area of a given bbox!'''
        return 2 * sum(map(lambda l: l[0] * l[1], itertools.combinations(bbox[1] - bbox[0], 2)))

    def uniform_boundary_points(self, n):
        """Uniform boundary points on the spatio-temporal domain.

        Geometry surface area ~ bbox.
        Time surface area ~ diam.
        """
        if self.geometry.dim == 1:
            nx = 2
        else:
            s = GeometryXTime._cube_surface_area(self.bbox)
            nx = int((n * s / self.timedomain.diam) ** 0.5)
        nt = int(np.ceil(n / nx))
        x = self.geometry.uniform_boundary_points(nx)
        nx = len(x)
        t = np.linspace(
            self.timedomain.t1,
            self.timedomain.t0,
            num=nt,
            endpoint=False)
        xt = []
        for ti in t:
            xt.append(np.hstack((x, np.full([nx, 1], ti))))
        xt = np.vstack(xt)
        if n != len(xt):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(xt))
            )
        return xt

    def random_boundary_points(self, n, random="pseudo"):
        x = self.geometry.random_boundary_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        t = np.random.default_rng().permutation(t)
        return np.hstack((x, t))

    def uniform_initial_points(self, n):
        x = self.geometry.uniform_points(n, True)
        t = self.timedomain.t0
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return np.hstack((x, np.full([len(x), 1], t)))

    def random_initial_points(self, n, random="pseudo"):
        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.t0
        return np.hstack((x, np.full([n, 1], t)))


class Disk(Geometry):
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
        super(Disk, self).__init__(
            2, (self.center - radius, self.center + radius), 2 * radius
        )

        self._r2 = radius ** 2

    def inside(self, x):
        return np.linalg.norm(x - self.center, axis=-1) <= self.radius

    def on_boundary(self, x):
        return np.isclose(np.linalg.norm(x - self.center, axis=-1), self.radius)

    def random_points(self, n, random="pseudo"):
        rng = sample(n, 2, random)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        x, y = np.cos(theta), np.sin(theta)
        return self.radius * (np.sqrt(r) * np.vstack((x, y))).T + self.center

    def uniform_boundary_points(self, n):
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        X = np.vstack((np.cos(theta), np.sin(theta))).T
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        theta = 2 * np.pi * u
        X = np.hstack((np.cos(theta), np.sin(theta)))
        return self.radius * X + self.center


class Hypercube(Geometry):
    def __init__(self, xmin, xmax):
        if len(xmin) != len(xmax):
            raise ValueError("Dimensions of xmin and xmax do not match.")
        if np.any(np.array(xmin) >= np.array(xmax)):
            raise ValueError("xmin can't be >=xmax")

        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.side_length = self.xmax - self.xmin
        super(Hypercube, self).__init__(
            len(xmin), (self.xmin, self.xmax), np.linalg.norm(self.side_length)
        )
        self.volume = np.prod(self.side_length)

    def inside(self, x):
        return np.logical_and(
            np.all(x >= self.xmin, axis=-1), np.all(x <= self.xmax, axis=-1)
        )

    def on_boundary(self, x):
        _on_boundary = np.logical_or(
            np.any(np.isclose(x, self.xmin), axis=-1),
            np.any(np.isclose(x, self.xmax), axis=-1),
        )
        return np.logical_and(self.inside(x), _on_boundary)

    def uniform_points(self, n, boundary=True):
        dx = (self.volume / n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = self.side_length[i] // dx
            if boundary:
                xi.append(
                    np.linspace(
                        self.xmin[i], self.xmax[i], num=ni,
                    )
                )
            else:
                xi.append(
                    np.linspace(
                        self.xmin[i],
                        self.xmax[i],
                        num=ni + 1,
                        endpoint=False,
                    )[1:]
                )
        x = np.array(list(itertools.product(*xi)))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_points(self, n, random="pseudo"):
        x = sample(n, self.dim, random)
        return (self.xmax - self.xmin) * x + self.xmin

    def random_boundary_points(self, n, random="pseudo"):
        x = sample(n, self.dim, random)

        rng = np.random.default_rng()
        rand_dim = rng.integers(self.dim, size=n)
        # Replace value of the randomly picked dimension with the nearest boundary value (0 or 1)
        x[np.arange(n), rand_dim] = np.round(x[np.arange(n), rand_dim])
        return (self.xmax - self.xmin) * x + self.xmin

class Rectangle(Hypercube):
    def __init__(self, xmin, xmax):
        super(Rectangle, self).__init__(xmin, xmax)
        self.perimeter = 2 * np.sum(self.xmax - self.xmin)
        self.area = np.prod(self.xmax - self.xmin)

class Ellipse(Geometry):
    """
    Args:
        center: Center of the ellipse.
        semimajor: Semimajor of the ellipse.
        semiminor: Semiminor of the ellipse.
        angle: Rotation angle of the ellipse. A positive angle rotates the ellipse
            clockwise about the center and a negative angle rotates the ellipse
            counterclockwise about the center.
    """

    def __init__(self, center, semimajor, semiminor, angle=0):
        self.center = np.array(center)
        self.semimajor = semimajor
        self.semiminor = semiminor
        self.angle = angle
        self.c = (semimajor**2 - semiminor**2) ** 0.5

        self.focus1 = np.array(
            [
                center[0] - self.c * np.cos(angle),
                center[1] + self.c * np.sin(angle),
            ],
        )
        self.focus2 = np.array(
            [
                center[0] + self.c * np.cos(angle),
                center[1] - self.c * np.sin(angle),
            ],
        )
        self.rotation_mat = np.array(
            [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
        )

        _, self.total_arc = self._theta_from_arc_length_constructor()

    def _ellipse_arc(self):

        theta = np.linspace(0, 2 * np.pi, 10000)
        coords = np.array(
            [self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)]
        )

        coords_diffs = np.diff(coords)

        delta_r = np.linalg.norm(coords_diffs, axis=0)
        cumulative_distance = np.concatenate(([0], np.cumsum(delta_r)))
        c = np.sum(delta_r)
        return theta, cumulative_distance, c

    def _theta_from_arc_length_constructor(self):
        theta, cumulative_distance, total_arc = self._ellipse_arc()

        def f(s):
            return np.interp(s, cumulative_distance, theta)

        return f, total_arc

    def on_boundary(self, x):
        d1 = np.linalg.norm(x - self.focus1, axis=-1)
        d2 = np.linalg.norm(x - self.focus2, axis=-1)
        return np.isclose(d1 + d2, 2 * self.semimajor)

    def inside(self, x):
        d1 = np.linalg.norm(x - self.focus1, axis=-1)
        d2 = np.linalg.norm(x - self.focus2, axis=-1)
        return d1 + d2 <= 2 * self.semimajor

    def random_points(self, n, random="pseudo"):
        rng = sample(n, 2, random)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        x, y = self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)
        X = np.sqrt(r) * np.vstack((x, y))
        return np.matmul(self.rotation_mat, X).T + self.center

    def uniform_boundary_points(self, n):
        u = np.linspace(0, 1, num=n, endpoint=False).reshape((-1, 1))
        theta = self.theta_from_arc_length(u * self.total_arc)
        X = np.hstack((self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)))
        return np.matmul(self.rotation_mat, X.T).T + self.center

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        theta = self.theta_from_arc_length(u * self.total_arc)
        X = np.hstack((self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)))
        return np.matmul(self.rotation_mat, X.T).T + self.center
