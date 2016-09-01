import pymc3.distributions as dist
import theano.tensor as tt
from theano import scan

from diploma.utils import lags


class ARIMA(dist.Continuous):
    """
        ARIMA(p, d, q) process
        y = D^d(Y)
        y_t = (\alpha_1*L + \dots + \alpha_p*L^p)yt + (1 + \beta_1*L + \dots + \beta_q*L^q)e_t

        Parameters
        ----------
        omega : distribution
            omega > 0, distribution for variance
        alpha : distribution
            alpha >= 0, distribution for autoregressive term
        beta : distribution
            beta >= 0, alpha + beta < 1, distribution for moving
            average term
        start_e : distribution
            distribution for initial errors in estimation
        """
    def __init__(self, alpha, beta, omega, p=1, d=0, q=1, start_e=None, *args, **kwargs):
        super(ARIMA, self).__init__(*args, **kwargs)
        self.mean = 0
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.p = p
        self.d = d
        self.q = q
        if start_e is None:
            import numpy as np
            self.evec = np.zeros(q)
        else:
            self.evec = start_e

    def logp(self, x):
        L = lags(x, k=self.p+1)
        dL = tt.extra_ops.diff(L, self.d, 0)
        y, X = dL[:,0], dL[:,1:]
        etaps = list(-i for i in range(1, self.q+1))

        def e_hat(y, X, *es):
            e = tt.stack(es[:-2])
            return y - tt.dot(X, es[-2]) - tt.dot(e, es[-1])

        e, _ = scan(fn=e_hat, sequences=[dict(input=y), dict(input=X)],
                    outputs_info=dict(initial=self.evec, taps=etaps),
                    non_sequences=[self.alpha, self.beta],
                    truncate_gradient=max(self.p,self.q))
        # I don't care about initial e yet
        return tt.sum(dist.Normal.dist(0, sd=self.omega).logp(e))