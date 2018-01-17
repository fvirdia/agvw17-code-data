# -*- coding: utf-8 -*-
"""
Parameter estimation for various schemes from the literature.
This script assumes that the LWE estimator commit cc5f6e8 is present in a subdirectory `estimator`.

AUTHOR:

    Martin R. Albrecht - 2017
    Fernando Virdia - 2017

"""
from sage.all import *
import newestimator
import oldestimator.estimator as oldestimator
from sage.all import sqrt, Infinity, log, ceil, ZZ, RR, pi

def mcost(cost, kwd):
    res = "%10s beta %3d, dim %4d, rop %6.2f, red %6.2f"%(kwd, cost[kwd]["beta"], cost[kwd]["d"], log(cost[kwd]["rop"],2).n(), log(cost[kwd]["red"], 2).n())
    return res

def Lizard():
    """
    EPRINT:CKLS16
    """

    def _lizard(n, alpha, q, h, asymptotic):
        reduction_cost_model = lambda beta, d, B: 2**asymptotic(beta, d)

        sd = alpha * q / sqrt(2*pi)
        small = sqrt(1.5)*sd # ternary secret
        sparse = sd

        kannan   = newestimator.primal_usvp(n, alpha, q, secret_distribution="normal",
                                         m=newestimator.oo,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)

        scaled   = newestimator.primal_usvp(n, alpha, q, secret_distribution=(-1, 1),
                                         m=newestimator.oo,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)

        weighted = newestimator.primal_usvp(n, alpha, q, secret_distribution=((-1, 1), h),
                                 m=newestimator.oo,  success_probability=0.99,
                                 reduction_cost_model=reduction_cost_model)

        primald = newestimator.partial(newestimator.drop_and_solve, newestimator.primal_usvp, postprocess=False, decision=False)
        dropped = primald(n, alpha, q, secret_distribution=((-1, 1), h),
                                         m=newestimator.oo,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)

        return {"kannan": kannan, "scaled": scaled, "weighted": weighted, "dropped": dropped}

    def _lizard_cost(n, alpha, q, h, proposed_b, label, asymptotic):
        cost = _lizard(n, alpha, q, h, asymptotic)
        print label
        print
        print "Standard deviation %.2f"%(alpha * q / sqrt(2*pi))
        print "Proposed beta %d log_T %.2f"%(proposed_b, asymptotic(proposed_b, -1))
        print mcost(cost, "kannan")
        print mcost(cost, "scaled")
        print mcost(cost, "weighted")
        print mcost(cost, "dropped")
        print

    print
    print "Lizard"
    print

    LIZARD = [(303, 1.0/419, 2**11, 64, 297, "Challenge", lambda k, n: RR(log(k, 2) + 0.292*k)),
              (386, 1.0/391, 2**11, 64, 418, "Classical", lambda k, n: RR(log(k, 2) + 0.292*k)),
              (414, 1.0/400, 2**11, 64, 456, "Quantum",   lambda k, n: RR(log(k, 2) + 0.265*k)),
              (504, 1.0/389, 2**12, 64, 590, "Paranoid",  lambda k, n: RR(log(k, 2) + 0.2075*k))]

    for i in range(len(LIZARD)):
        n, alpha, q, h, proposed_b, lable, asymptotic = LIZARD[i]
        _lizard_cost(n, alpha, q, h, proposed_b, lable, asymptotic)


def HELib():
    """
    EPRINT:GenHalSma12
    EC:Albrecht17
    """

    def _helib(n, sd, q, h, asymptotic):
        reduction_cost_model = lambda beta, d, B: 2**asymptotic(beta, d)

        alpha = RR(sqrt(2*pi)*sd/q)
        small = sqrt(1.5)*sd # ternary
        sparse = sd

        kannan = newestimator.primal_usvp(n, alpha, q, secret_distribution="normal",
                                         m=newestimator.oo,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)

        scaled = newestimator.primal_usvp(n, alpha, q, secret_distribution=(-1, 1),
                                         m=newestimator.oo,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)

        weighted = newestimator.primal_usvp(n, alpha, q, secret_distribution=((-1, 1), h),
                                         m=newestimator.oo,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)

        primald = newestimator.partial(newestimator.drop_and_solve, newestimator.primal_usvp, postprocess=False, decision=False)
        dropped = primald(n, alpha, q, secret_distribution=((-1, 1), h),
                                         m=newestimator.oo,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)

        return {"kannan": kannan, "scaled": scaled, "weighted": weighted, "dropped": dropped}

    def _helib_cost(n, sd, q, h, proposed_log_T, label, asymptotic):
        cost = _helib(n, sd, q, h, asymptotic)
        print label
        print
        print "Standard deviation %.2f"%(sd)
        print "Proposed beta %d log_T %.2f"%(-1, proposed_log_T)
        print mcost(cost, "kannan")
        print mcost(cost, "scaled")
        print mcost(cost, "weighted")
        print mcost(cost, "dropped")
        print

    print
    print "HELib"
    print

    HELIB = [(1024, 3.2, 2**47, 64, 80),
             (2048, 3.2, 2**87, 64, 80),
             (4096, 3.2, 2**167, 64, 80),
             (8192, 3.2, 2**326, 64, 80),
             (16384, 3.2, 2**638, 64, 80),
             (1024, 3.2, 2**38, 64, 128),
             (2048, 3.2, 2**70, 64, 128),
             (4096, 3.2, 2**134, 64, 128),
             (8192, 3.2, 2**261, 64, 128),
             (16384, 3.2, 2**511, 64, 128)]

    for i in range(len(HELIB)):
        n, sd, q, h, proposed_log_T = HELIB[i]
        _helib_cost(n, sd, q, h, proposed_log_T, "n %d - %d bit"%(n, proposed_log_T),
                    oldestimator.bkz_runtime_k_sieve_bdgl16_asymptotic)


def SEAL():
    """
    EPRINT:LaiChePla16
    EC:Albrecht17
    """

    def _seal(n, sd, q, asymptotic):
        optimisation_target = "sieve"
        oldestimator.bkz_runtime_k_sieve_asymptotic = asymptotic
        reduction_cost_model = lambda beta, d, B: 2**asymptotic(beta, d)


        alpha = RR(sqrt(2*pi)*sd/q)
        small = sqrt(1.5)*sd  # ternary

        proposed = oldestimator.sis_small_secret_mod_switch(n=n, q=q, alpha=alpha, secret_bounds=(-1, 1), use_lll=True,
                                                   optimisation_target=optimisation_target)

        kannan  = newestimator.primal_usvp(n, alpha, q, secret_distribution="normal",
                                         m=newestimator.oo,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)

        scaled  = newestimator.primal_usvp(n, alpha, q, secret_distribution=(-1, 1),
                                         m=newestimator.oo,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)

        return {"kannan": kannan, "scaled": scaled, "proposed": proposed}

    def _seal_cost(n, sd, q, label, asymptotic):
        cost = _seal(n, sd, q, asymptotic)
        print label
        print
        print "Standard deviation %.2f"%(sd)
        print "Proposed beta %d m %d log_T %.2f"%(cost["proposed"]["beta"], cost["proposed"]["dim"], log(cost["proposed"]["sieve"], 2))
        print mcost(cost, "kannan")
        print mcost(cost, "scaled")
        print

    print
    print "SEAL"
    print

    SEAL = [(1024, 3.19, 2**35 - 2**14 + 2**11 + 1),
            (2048, 3.19, 2**60 - 2**14 + 1),
            (4096, 3.19, 2**116 - 2**18 + 1),
            (8192, 3.19, 2**226 - 2**26 + 1),
            (16384, 3.19, 2**435 - 2**33 + 1), ]

    for i in range(len(SEAL)):
        n, sd, q = SEAL[i]
        _seal_cost(n, sd, q, "n %d classic"%n, oldestimator.bkz_runtime_k_sieve_bdgl16_asymptotic)


def Tesla():
    """
    PQ:ABBDEGKP17
    """
    def _tesla(n, sd, q, m, asymptotic):
        optimisation_target = "sieve"
        oldestimator.bkz_runtime_k_sieve_asymptotic = asymptotic
        reduction_cost_model = lambda beta, d, B: 2**asymptotic(beta, d)
        alpha = RR(sd)/q*sqrt(2*pi)

        old = oldestimator.kannan(n, alpha, q, optimisation_target=optimisation_target, samples=m)
        new = newestimator.primal_usvp(n, alpha, q, secret_distribution="normal",
                                         m=m,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)
        return {"old": old, "kannan": new}

    def _tesla_cost(n, sd, q, m, label, asymptotic):
        cost = _tesla(n, sd, q, m, asymptotic)
        print label
        print
        print "Standard deviation %.2f"%(sd)
        print "Proposed beta %3d        log_T %.2f"%(cost["old"]["beta"], log(cost["old"]["sieve"], 2))
        print mcost(cost, "kannan")
        print

    print
    print "TESLA"
    print

    def _bkz_runtime_k_enum_quantum(k, n):
        repeat = log(oldestimator.bkz_svp_repeat(n, k), 2)
        return RR((log(k, 2)*k*0.0225 + 0.4574*k - k/RR(4))/RR(2) + repeat)

    TESLA = [(644,  55, 2**31 - 99, 3156),
             (804,  57, 2**31 - 19, 4972),
             (1300, 73, 40582171961, 4788)]

    for i in range(len(TESLA)):
        n, sd, q, m = TESLA[i]
        _tesla_cost(n, sd, q, m, "TESLA-%d"%i, asymptotic=oldestimator.bkz_runtime_k_sieve_bdgl16_asymptotic)

    for i in range(len(TESLA)):
        n, sd, q, m = TESLA[i]
        _tesla_cost(n, sd, q, m, "TESLA-%d"%i, asymptotic=_bkz_runtime_k_enum_quantum)


def FVSHE():
    """
    EPRINT:BCIV16
    """

    def _fvshe(n, sd, q, asymptotic):
        reduction_cost_model = lambda beta, d, B: 2**asymptotic(beta, d)

        alpha = RR(sqrt(2*pi)*sd/q)
        small = sqrt(1.5)*sd # ternary

        kannan = newestimator.primal_usvp(n, alpha, q, secret_distribution="normal",
                                         m=newestimator.oo,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)

        scaled = newestimator.primal_usvp(n, alpha, q, secret_distribution=(-1, 1),
                                         m=newestimator.oo,  success_probability=0.99,
                                         reduction_cost_model=reduction_cost_model)
        return {"kannan": kannan, "scaled": scaled}

    def _fvshe_cost(n, sd, q, label, asymptotic):
        cost = _fvshe(n, sd, q, asymptotic)
        print label
        print
        print "Standard deviation %.2f"%(sd)
        print mcost(cost, "kannan")
        print mcost(cost, "scaled")
        print

    print
    print "Smart Grid"
    print

    n, sd, q = 4096, 102, 2**186
    _fvshe_cost(n, sd, q, "80 bit classic security", oldestimator.bkz_runtime_k_sieve_bdgl16_asymptotic)
