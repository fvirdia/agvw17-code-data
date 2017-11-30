# -*- coding: utf-8 -*-
"""
Implementation of Algorithm 2 in [AGVW17].

AUTHOR:

    Martin R. Albrecht - 2017
    Fernando Virdia - 2017

REFERENCES:

    [AGVW17] Albrecht, Martin R., Florian Göpfert, Fernando Virdia,
    and Thomas Wunderer. "Revisiting the Expected Cost of Solving uSVP and
    Applications to LWE." ASIACRYPT, 2017. https://ia.cr/2017/815
"""
from collections import OrderedDict
from sage.all import vector, RR, log
from fpylll.algorithms.bkz import BKZReduction as BKZBase
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.algorithms.bkz_stats import BKZTreeTracer, dummy_tracer, Statistic, pretty_dict
from fpylll import BKZ, Enumeration, EnumerationError
from fpylll.util import adjust_radius_to_gh_bound
import time
import sys


class TrackSuccess(BKZTreeTracer):
    def __init__(self, *args, **kwds):
        self.evc = kwds.pop("evc", None)
        self.block_size = kwds.pop("block_size", None)
        BKZTreeTracer.__init__(self, *args, **kwds)
        self.stats = {
            "success": False,
            "cputime": None,
            "proj": [],
            "vec": [],
            "e_recovered_by_last_size_reduction": False,
            "plots": {}
        }

        self.bound = 1.1*self.evc.norm()
        self.e_in_basis = False
        # start recording when d-kappa <= from_the_end
        self.from_the_end = 2*self.block_size

    def enter(self, label, **kwds):
        BKZTreeTracer.enter(self, label, **kwds)
        if label == "get_projections":
            M = kwds.get("M", None)
            stage = kwds.get("stage", None)
            tour_n = kwds.get("tour_n", None)
            kappa = kwds.get("kappa", None)

            if stage == "proj":
                # find projection index and record it
                ind = self._fast_e_star_index(M, self.evc)
                self.stats["proj"].append((tour_n, kappa, ind))

                # check if e already in basis.
                # If so, it's not merit of size_reduction from the previous step
                (e_in_basis, j) = self._check_e_in_basis(M)
                if e_in_basis:
                    self.e_in_basis = True

            if stage == "vec":
                # since this line of code is being executed,
                # last size_reduction was not the final one
                self.stats["e_recovered_by_last_size_reduction"] = False

                # check if error vector in basis
                (e_in_basis, j) = self._check_e_in_basis(M)

                if e_in_basis:
                    if not self.e_in_basis:
                        # it was this size_reduction to recover
                        # if any other SR happen, this is reset
                        self.stats["e_recovered_by_last_size_reduction"] = True
                    else:
                        # e was already in basis, so we leave as "not rec by last size_reduction".
                        pass
                    # add position of e in basis to record, it should match j+1
                    ind = self._fast_e_star_index(M, self.evc)
                    self.stats["vec"].append((tour_n, kappa, j+1, ind, map(int, list(M.B[j]))))
                    self.e_in_basis = True
                    self.stats["success"] = True
        elif label == "plots":
            M = kwds.get("M", None)
            tour_n = kwds.get("tour_n", None)
            kappa = kwds.get("kappa", None)

            # here goes the plotting
            if tour_n not in self.stats["plots"]:
                self.stats["plots"][tour_n] = {}

            self.stats["plots"][tour_n][kappa] = {}
            self.stats["plots"][tour_n][kappa]["gso_norms"] = [RR(log(M.get_r(i, i), 2)) for i in range(M.d)]
            # uncomment to keep track of ||v_i^*||, very slow
            # evc_vects = [self._proj(M, self.evc, i) for i in range(M.d)]
            # self.stats["plots"][tour_n][kappa]["evc_norms"] = map(lambda x: RR(2*log(x.norm(), 2)), evc_vects)

    def _check_e_in_basis(self, M):
        e_in_basis = False
        j = -1
        for i in range(M.d):
            if M.B[i].norm() < self.bound and M.B[i].norm() > 0:
                j = i
                e_in_basis = True
        return (e_in_basis, j)

    def _fast_e_star_index(self, M, e):
        M.update_gso()
        if len(e) == M.d - 1:
            e = vector(list(e) + [1])

        v = M.from_canonical(e)

        for i in range(len(v))[::-1]:
            if abs(v[i]) > 0.01:
                return i+1
        else:
            raise ValueError("The error vector appears to be 0")

    def _proj(self, M, v, i):
        if i == 0:
            return v
        return v - vector(RR, M.to_canonical(list(M.from_canonical(v, 0, i))))

    def exit(self, **kwds):
        """
        By default CPU and wall time are recorded.  More information is recorded for "enumeration"
        and "tour" labels.  When the label is a tour then the status is printed if verbosity > 0.
        """
        node = self.current
        label = node.label

        # only change is here adding a try/except
        try:
            node.data["cputime"] += time.clock()
            node.data["walltime"] += time.time()
        except KeyError:
            pass

        if label == "enumeration":
            full = kwds.get("full", True)
            if full:
                node.data["#enum"] = Statistic(kwds["enum_obj"].get_nodes(), repr="sum") + node.data.get("#enum", None)
                try:
                    node.data["%"] = Statistic(kwds["probability"], repr="avg") + node.data.get("%", None)
                except KeyError:
                    pass

        if label[0] == "tour":
            node.data["r_0"] = Statistic(self.instance.M.get_r(0, 0), repr="min")
            node.data["/"] = Statistic(self.instance.M.get_current_slope(0, self.instance.A.nrows), repr="min")

        if self.verbosity and label[0] == "tour":
            report = OrderedDict()
            report["i"] = label[1]
            report["cputime"] = node["cputime"]
            report["walltime"] = node["walltime"]
            try:
                report["preproc"] = node.find("preprocessing", True)["cputime"]
            except KeyError:
                pass
            try:
                report["svp"] = node.find("enumeration", True)["cputime"]
            except KeyError:
                pass
            report["lll"] = node.sum("cputime", label="lll")
            try:
                report["postproc"] = node.find("postprocessing", True)["cputime"]
            except KeyError:
                pass
            try:
                report["pruner"] = node.find("pruner", True)["cputime"]
            except KeyError:
                pass
            report["r_0"] = node["r_0"]
            report["/"] = node["/"]
            report["#enum"] = node.sum("#enum")

            print(pretty_dict(report))

        self.current = self.current.parent


class StatBKZ(BKZReduction):
    def __call__(self, params, tracer, min_row=0, max_row=-1, plots=False):
        """Run the BKZ algorithm with parameters `param`.

        :param params: BKZ parameters
        :param min_row: start processing in this row
        :param max_row: stop processing in this row (exclusive)
        :param plots: if True, calls tracker to generate plots

        """

        self.plots = plots

        if params.flags & BKZ.AUTO_ABORT:
            auto_abort = BKZ.AutoAbort(self.M, self.A.nrows)

        cputime_start = time.clock()

        with tracer.context("lll"):
            self.lll_obj()

        print
        print "block size:  %s, flags: %s, max_loops:   %s, max_time: %s, autoAbort: %s,"%(params.block_size,
                                                                                           oct(params.flags),
                                                                                           params.max_loops,
                                                                                           params.max_time,
                                                                                           params.auto_abort)

        i = 0
        while True:
            with tracer.context("tour", i):
                time_delta = time.time()
                print "tour", i
                sys.stdout.flush()
                clean = self.tour(params, min_row, max_row, tracer, called_from_call=True, tour_n=i)
                time_delta = time.time() - time_delta
                print "End of BKZ tour    %s, time =    %s"%(i, time_delta)
            i += 1

            if clean:
                break
            if params.block_size >= self.A.nrows:
                break
            if (params.flags & BKZ.AUTO_ABORT) and auto_abort.test_abort():
                break
            if (params.flags & BKZ.MAX_LOOPS) and i >= params.max_loops:
                break
            if (params.flags & BKZ.MAX_TIME) and time.clock() - cputime_start >= params.max_time:
                break

        tracer.exit()
        self.trace = tracer.trace
        return clean

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer, called_from_call=False, tour_n=None):
        """One BKZ loop over all indices.

        :param params: BKZ parameters
        :param min_row: start index ≥ 0
        :param max_row: last index ≤ n

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        if max_row == -1:
            max_row = self.A.nrows

        clean = True

        i = 0
        for kappa in range(min_row, max_row-2):
            i += 1
            block_size = min(params.block_size, max_row - kappa)
            with tracer.context('kappa', i):
                clean &= self.svp_reduction(kappa, block_size, params, tracer,
                                            should_I_log=called_from_call,
                                            from_the_end=max_row-kappa,
                                            tour_n=tour_n)

                if tracer.e_in_basis:
                    # terminate BKZ
                    return True

        return clean

    def svp_postprocessing(self, kappa, block_size, solution, tracer):
        """Insert SVP solution into basis and LLL reduce.

        :param solution: coordinates of an SVP solution
        :param kappa: current index
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        if solution is None:
            return True

        nonzero_vectors = len([x for x in solution if x])
        if nonzero_vectors == 1:
            first_nonzero_vector = None
            for i in range(block_size):
                if abs(solution[i]) == 1:
                    first_nonzero_vector = i
                    break

            self.M.move_row(kappa + first_nonzero_vector, kappa)
            with tracer.context("lll"):
                self.lll_obj.size_reduction(kappa, kappa + first_nonzero_vector + 1, kappa)

        else:
            d = self.M.d
            self.M.create_row()

            with self.M.row_ops(d, d+1):
                for i in range(block_size):
                    self.M.row_addmul(d, kappa + i, solution[i])

            self.M.move_row(d, kappa)
            with tracer.context("lll"):
                self.lll_obj(kappa, kappa, kappa + block_size + 1, kappa)

            self.M.move_row(kappa + block_size, d)
            self.M.remove_last_row()

        return False

    def svp_preprocessing(self, kappa, block_size, param, tracer=dummy_tracer):
        clean = True

        lll_start = kappa if param.flags & BKZ.BOUNDED_LLL else 0
        with tracer.context("lll"):
            self.lll_obj(lll_start, lll_start, kappa + block_size, lll_start)
            if self.lll_obj.nswaps > 0:
                clean = False

        clean &= BKZBase.svp_preprocessing(self, kappa, block_size, param, tracer)

        for preproc in param.strategies[block_size].preprocessing_block_sizes:
            prepar = param.__class__(block_size=preproc, strategies=param.strategies, flags=BKZ.GH_BND | BKZ.BOUNDED_LLL)
            clean &= self.tour(prepar, kappa, kappa + block_size, tracer=dummy_tracer)

        return clean

    def svp_reduction(self, kappa, block_size, param, tracer=dummy_tracer, should_I_log=False, tour_n=None, from_the_end=None):
        """

        :param kappa:
        :param block_size:
        :param params:
        :param tracer:

        """
        # Dummy Tracer
        if not hasattr(tracer, "evc"):
            tracer.from_the_end = -1
            tracer.e_in_basis = False

        self.lll_obj.size_reduction(0, kappa+1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)

        remaining_probability, rerandomize = 1.0, False

        while remaining_probability > 1. - param.min_success_probability:
            with tracer.context("preprocessing"):
                if rerandomize:
                    with tracer.context("randomization"):
                        self.randomize_block(kappa+1, kappa+block_size,
                                             density=param.rerandomization_density, tracer=tracer)
                with tracer.context("reduction"):
                        self.svp_preprocessing(kappa, block_size, param, tracer=tracer)

            radius, expo = self.M.get_r_exp(kappa, kappa)
            radius *= self.lll_obj.delta

            root_det = self.M.get_root_det(kappa, kappa + block_size)
            radius, expo = adjust_radius_to_gh_bound(radius, expo, block_size, root_det, param.gh_factor)
            gh = sys.float_info.max
            (gh, _) = adjust_radius_to_gh_bound(gh, expo, block_size, root_det, param.gh_factor)

            pruning = self.get_pruning(kappa, block_size, param, tracer)

            try:
                enum_obj = Enumeration(self.M)
                with tracer.context("enumeration",
                                    enum_obj=enum_obj,
                                    probability=pruning.expectation,
                                    full=block_size == param.block_size):
                    solution, max_dist = enum_obj.enumerate(kappa, kappa + block_size, radius, expo,
                                                            pruning=pruning.coefficients)[0]

                with tracer.context("postprocessing",
                                    block_size=block_size,
                                    kappa=kappa,
                                    solution=solution,
                                    max_dist=max_dist,
                                    gaussian=(gh, expo)):
                    self.svp_postprocessing(kappa, block_size, solution, tracer=tracer)
                rerandomize = False

            except EnumerationError:
                rerandomize = True

            remaining_probability *= (1 - pruning.expectation)

        # record position of projection of (e|1)
        if from_the_end <= tracer.from_the_end and should_I_log:
            with tracer.context("get_projections",
                                M=self.M,
                                stage="proj",
                                tour_n=tour_n,
                                kappa=kappa):
                pass

        self.lll_obj.size_reduction(0, kappa+1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        # check if (e|1) in basis
        if from_the_end <= tracer.from_the_end and should_I_log:
            with tracer.context("get_projections",
                                M=self.M,
                                stage="vec",
                                tour_n=tour_n,
                                kappa=kappa):
                pass

        # preparing plots
        if self.plots:
            with tracer.context("plots",
                                M=self.M,
                                tour_n=tour_n,
                                kappa=kappa):
                pass

        clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        return clean
