import tensorflow as tf
from baselines.common import zipsame
import numpy as np
from baselines import logger
from .net_utils import timed, intprod, GetFlat, SetFromFlat, function, flatgrad
from baselines.common.cg import cg

class TrpoSolver(object):

    def __init__(self, sess, pi_pd, oldpi_pd, image_input,speed_input, ac, var_list, entcoeff):

        self.sess = sess
        self.atarg = tf.placeholder(dtype=tf.float32, shape=[None])
        self.kloldnew = oldpi_pd.kl(pi_pd)
        self.meankl = tf.reduce_mean(self.kloldnew)
        self.ent = pi_pd.entropy()
        self.meanent = tf.reduce_mean(self.ent)
        self.meanent = tf.stop_gradient(self.meanent)
        self.entbonus = entcoeff * self.meanent
        self.ac = ac
        self.ratio = tf.exp(pi_pd.logp(self.ac) - oldpi_pd.logp(self.ac))  # advantage * pnew / pold
        self.surrgain = tf.reduce_mean(self.ratio * self.atarg)
        self.optimgain = self.surrgain + self.entbonus
        self.losses = [self.optimgain, self.meankl, self.entbonus, self.surrgain, self.meanent]

        self.dist = self.meankl
        self.get_flat = GetFlat(var_list, sess)
        self.set_from_flat = SetFromFlat(var_list, sess)
        self.klgrads = tf.gradients(self.dist, var_list)
        self.flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        self.shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        self.tangents = []
        for shape in self.shapes:
            sz = intprod(shape)
            self.tangents.append(tf.reshape(self.flat_tangent[start:start + sz], shape))
            start += sz
        self.gvp = tf.add_n(
            [tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(self.klgrads, self.tangents)])  # pylint: disable=E1111
        self.fvp = flatgrad(self.gvp, var_list)

        self.compute_losses = function([image_input, speed_input, self.ac, self.atarg], self.losses, sess)
        self.compute_lossandgrad = function([image_input, speed_input, self.ac, self.atarg], self.losses + [flatgrad(self.optimgain, var_list)], sess)
        self.compute_fvp = function([self.flat_tangent, image_input, speed_input, self.ac, self.atarg], self.fvp, sess)

    def run(self, args, assign_old_eq_new, cg_damping, max_kl, cg_iters):
        fvpargs = [arr[::5] for arr in args]

        def fisher_vector_product(p):
            return self.compute_fvp(p, *fvpargs) + cg_damping * p

        with timed("computegrad"):
            *lossbefore, g = self.compute_lossandgrad(*args)
            lossbefore = np.array(lossbefore)

        assign_old_eq_new()


        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose= True)
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = self.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self.set_from_flat(thnew)
                surr, kl, *_ = np.array(self.compute_losses(*args))
                meanlosses = surr
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                self.set_from_flat(thbefore)


        return meanlosses