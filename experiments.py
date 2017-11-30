# -*- coding: utf-8 -*-
"""
Code necessary to reproduce and analyse the results from [AGVW17].

AUTHOR:

    Fernando Virdia - 2017

REFERENCES:

    [AGVW17] Albrecht, Martin R., Florian Göpfert, Fernando Virdia,
    and Thomas Wunderer. "Revisiting the Expected Cost of Solving uSVP and
    Applications to LWE." ASIACRYPT, 2017. https://ia.cr/2017/815
"""

from sage.all import *
from sage.crypto.lwe import LWE, DiscreteGaussianDistributionIntegerSampler
import fpylll
from fpylll.fplll.bkz_param import BKZParam
from StatBKZ import StatBKZ, TrackSuccess
import random
import time
import sys

experiment_parameters = {
    #        seed,        n,                 q,             sd,   m, bs, float_type, mpfr_precision, max_tours,                                                          save_proc, plots
    "n65":  [[seed + 1,  65,  next_prime(2**9), 8.0/sqrt(2*pi), 182, 56,        "d",           None,        20,                                                               None, False] for seed in range(10000)],
    "n80":  [[seed + 1,  80, next_prime(2**10), 8.0/sqrt(2*pi), 204, 60,       "dd",           None,        20,                                                               None, False] for seed in range(1000)],
    "n100": [[seed + 1, 100,              2053, 8.0/sqrt(2*pi), 243, 67,       "dd",           None,        20, lambda ret, seed, b: save(ret, "experiments/n100b%d/%d"%(b, seed)), False] for seed in range(500)],
    "n108": [[seed + 1, 108,              2053, 8.0/sqrt(2*pi), 261, 77,       "dd",           None,        40,  lambda ret, seed, b: save(ret, "experiments/n108_%d.sobj"%(seed)), False] for seed in range(5)],
    "n110": [[seed + 1, 110,              2053, 8.0/sqrt(2*pi), 272, 78,       "dd",           None,        40,  lambda ret, seed, b: save(ret, "experiments/n110_%d.sobj"%(seed)), False] for seed in range(5)]
}


def genLWEInstance(n,
                   q,
                   sd,
                   m,
                   float_type="d",
                   mpfr_precision=None):
    """Generate lattices from LWE instances using Kannan's embedding.

    :param n:                   secret dimension
    :param q:                   lwe modulo
    :param sd:                  standard deviation
    :param m:                   number of lwe samples
    :param float_type:          floating point type
    :param mpfr_precision:      floating point precision (if using mpfr)

    :returns:                   the lwe generator, the samples, the lattice and
                                its volume
    """
    # generate LWE instance
    lwe = LWE(n=n, q=q, D=DiscreteGaussianDistributionIntegerSampler(sd))

    # get m different LWE samples
    samples = [lwe() for i in range(m)]

    # generate kannan's embedding lattice
    A = matrix([a for a, c in samples])
    C = matrix(ZZ, 1, m, [c for a, c in samples])
    AT = A.T.echelon_form()
    B = AT.change_ring(ZZ).stack(matrix(m-n, n).augment(q*identity_matrix(m-n)))

    # embed the ciphertext to the lattice, so that error vector
    # becomes the (most likely unique) SVP in the lattice
    BC = B.stack(matrix(C).change_ring(ZZ))
    BC = BC.augment(matrix(m+1, 1))
    BC[-1, -1] = 1 # use 1 rather than ||e|| as embedding factor
    BC = fpylll.IntegerMatrix.from_matrix(BC)

    # preprocess basis
    fpylll.LLL.reduction(BC)

    # set floating point precision
    BC_GSO = fpylll.GSO.Mat(BC, float_type=float_type)
    if float_type == "mpfr":
        set_precision(mpfr_precision)
    BC_GSO.update_gso()

    # get lattice volume
    vol = sqrt(prod([RR(BC_GSO.get_r(i, i)) for i in range(m+1)]))

    return (lwe, samples, A, C, BC_GSO, vol)


def runBKZ(L,
           b,
           max_tours,
           evc=None,
           plots=False):

    """Set up and run Algorithm 2 from the paper, recording detailed statistics.

    :param L:           lattice basis
    :param b:           BKZ block size
    :param max_tours:   max number of BKZ tours
    :param evc:         target (uSVP) error vector
    :param plots:       if True, generate length plots for each BKZ step

    :returns:           the BKZ object and the tracer containing statistics
    """

    # set up BKZ
    params = BKZParam(block_size=b,
                      strategies=fpylll.BKZ.DEFAULT_STRATEGY,
                      flags=fpylll.BKZ.AUTO_ABORT
                            | fpylll.BKZ.GH_BND
                            | fpylll.BKZ.MAX_LOOPS
                            | fpylll.BKZ.BOUNDED_LLL,
                      max_loops=max_tours)
    bkz = StatBKZ(L)
    tracer = TrackSuccess(bkz, verbosity=False, evc=evc, block_size=b)
    bkz(params, tracer, plots=plots)
    return (bkz, tracer)


def run_experiment(n,
                   q,
                   sd,
                   m,
                   b,
                   float_type="d",
                   mpfr_precision=None,
                   max_tours=20,
                   seed=None,
                   verbose=False,
                   plots=False):
    """Generates an LWE instance, calls BKZ, tides up statistics and returns them.

    :param n:               secret dimension
    :param q:               lwe modulo
    :param sd:              standard deviation
    :param m:               number of lwe samples
    :param b:               bkz block size
    :param max_tours:       maximum number of BKZ tours to allow
    :param float_type:      floating point type
    :param mpfr_precision:  floating point precision (if using mpfr)
    :param seed:            seed for all the corresponding PNRGs
    :param verbose:         if True, print some extra information
    :param plots:           if True, produce plots for each BKZ step

    :returns:
    """

    # prepare seed and prngs
    if not seed:
        seed = ceil(10000000000*random())

    seed = int(seed) # serialization safe
    set_random_seed(seed) # sets sage prng
    fpylll.set_random_seed(seed) # sets underlying fplll prng
    random.seed(seed) # used by fpylll pruning
    if verbose:
        print "Seed", seed

    # generate lwe instance
    (oracle, samples, A, C, BC, vol) = genLWEInstance(n,
                                                      q,
                                                      sd,
                                                      m,
                                                      float_type=float_type,
                                                      mpfr_precision=mpfr_precision)

    # calculate the error vector
    evc = (C[0] - A*oracle._LWE__s) % q
    evc = vector(evc.list()+[1])
    evc = vector(map(lambda x: ZZ(x) if ZZ(x)< q//2 else ZZ(x)-q, evc))

    if verbose:
        print evc

    # run BKZ
    dim = m+1
    if verbose:
        print "Blocksize %d Samples %d Dimension %d"%(b,m,BC.d)

    cputime = time.clock()
    (bkz, tracer) = runBKZ(BC,
                           b,
                           max_tours,
                           evc=evc,
                           plots=plots)

    cputime = time.clock() - cputime

    # add some metadata to the statistics
    tracer.stats["cputime"] = cputime
    tracer.stats["seed"] = seed

    # show some stats
    if verbose:
        print "Success", tracer.stats["success"]
        print tracer.stats

    return tracer.stats


def singleThreadExperiment(seed,
                           n,
                           q,
                           sd,
                           m,
                           b,
                           float_type="d",
                           mpfr_precision=None,
                           max_tours=20,
                           save_proc=None,
                           plots=False):
    """Launch a single, monothread experiment

    :param seed:            seed for all the corresponding PNRGs
    :param n:               secret dimension
    :param q:               lwe modulo
    :param sd:              standard deviation
    :param m:               number of lwe samples
    :param b:               bkz block size
    :param float_type:      floating point type
    :param mpfr_precision:  floating point precision (if using mpfr)
    :param max_tours:       maximum number of BKZ tours to allow
    :param save_proc:       procedure to call after finishing the experiment to
                            save the results. If None, then the result is simply
                            returned but __not__ saved. Useful when generating
                            many instances with large returned statistics objects
    :param plots:           if True, generate plots of each BKZ step

    :returns:
    """

    try:
        ret = run_experiment(n,
                             q,
                             sd,
                             m,
                             b,
                             max_tours=max_tours,
                             float_type=float_type,
                             mpfr_precision=mpfr_precision,
                             seed=seed,
                             verbose=True,
                             plots=plots)
    except Exception, e:
        ret = str(e)
    finally:
        if save_proc:
            save_proc(ret, seed, b)
            return None
        else:
            return ret


@parallel(ncpus=42)
def parallelExperiment(batch):
    """Run in parallel a batch of single thread experiments
    :param batch:   an entry from the experiment_parameters dictionary

    :returns:       return values from the batch, depends on the batch's save_proc

    Example:
    >>> ret = parallelExperiment(experiment_parameters["n65"])
    >>> print ret
    >>> save(ret, "experiment_results.sobj")
    """
    seed, n, q, sd, m, b, float_type, mpfr_precision, max_tours, save_proc, plots = batch
    if not save_proc:
        print "Remember to store the returned value and save it!"
        print

    return singleThreadExperiment(seed,
                                  n,
                                  q,
                                  sd,
                                  m,
                                  b,
                                  float_type=float_type,
                                  mpfr_precision=mpfr_precision,
                                  max_tours=max_tours,
                                  save_proc=save_proc,
                                  plots=plots)


def extractExperimentResults(returns,
                             bs,
                             dim,
                             label="some label"):

    """Analyse returned/saved data from parallel experiments batches.
       It extracts the information needed for reproducing Table 1, except
       'same step' column.

       :param returns:      Parallel batch output
       :param bs:           Block size used
       :param dim:          Lattice dimension (m+1)
       :param label:        Label used for the batch. Used in filename for the pmf
                            plot generated (Figure 4).

       :returns:            Object containing the extracted information
    """

    # count experiments
    total = len(returns)

    # set parameters
    cutoff = dim-bs*0.9                     # consider also the first few blocks smaller than b

    # prepare lists
    recover = []                            # error vector was recovered
    not_recover = []                        # error vector wasn't recovered
    terminated = []                         # experiment terminated correctly, either recovering or not the error vector
    not_terminated = []                     # experiment failed because of bug (say ZeroDivisionError)

    recovered_late = []                     # anomalous instances were size_reduction works very late

    recovered_early = []                    # not late

    not_last_size_reduction = []            # e not recovered by last size reduction! ignore

    found_many_e = []                       # it seems many short vectors found;
                                            # should be empty

    proj_then_e = []                        # e is found in the same (tour, kappa) as \pi(e);
                                            # should be 99.99% of them

    not_proj_then_e = []                    # e is found at a (tour, kappa) where \pi(e) was not found
                                            # should be empty. Keep in mind late projectins

    not_proj_found_but_recover = []         # e was recover but \pi(e) swas never seen;
                                            # should be empty

    not_insta = []                          # multiple \pi(e) were found but e not recovered immediately
                                            # should be empty
    not_insta_location = []                 # check where did size_reduction fail

    not_rec_but_early_proj = []             # \pi(e) was found at one point, but e was never recovered
                                            # should be empty. Keep in mind late projectins

    pmf_k_where_e_is_recovered = [0]*dim    # it does not distinguish which were recovered by size_reduction and which not
    pmf_mean = 0
    pmf_tikz = ""

    cputime = 0                             # seconds used for completing all the experiments

    # start reading the records
    for x in returns:
        seed = x[0][0][0]
        inst = x[1]

        # if experiment terminated
        if 'vec' in inst:

            # add cputime
            cputime += RR(inst["cputime"])

            # add to list of terminated experiments
            terminated.append(seed)

            # error vector was found
            if len(inst['vec']) > 0:

                # add to list of recovered
                recover.append(seed)

                # found multiple e
                if len(inst['vec']) > 1:
                    found_many_e.append(seed)

                # get info about last step
                e_info = inst['vec'][0]
                found_e = e_info[:2] # = (tour_n, kappa)

                # save location where e found
                pmf_k_where_e_is_recovered[e_info[1]] += 1

                # were we expecting size_reduction to work?
                # not if kappa >= cutoff or e_index >= cutoff
                if found_e[1] >= cutoff or e_info[2] >= cutoff:
                    recovered_late.append(seed)

                # we recovered e where expected, let's see what happened with the projections
                else:
                    # since not late
                    recovered_early.append(seed)

                    # was it not size reduction?
                    if 'e_recovered_by_last_size_reduction' in inst:
                        if not inst['e_recovered_by_last_size_reduction']:
                            not_last_size_reduction.append(seed)


                    # get info about found projections
                    found_proj = inst['proj'] # = [(tour_n, kappa, e_index), ...]

                    # if no projections recorded
                    if len(found_proj) < 1:
                        not_proj_found_but_recover.append(seed)

                    # at least one projection recorded
                    else:
                        # last projection recorded at same (tour, kappa) were e was recoered
                        if found_proj[-1][:2] == found_e:

                            # proj -> recover
                            proj_then_e.append(seed)

                            # check earlier projections
                            for p in found_proj[:-1]:
                                # pi(e) found at an early kappa, and in a position where size_reduction will consider it, so <= kappa
                                if p[1] < cutoff and p[2] <= p[1]:
                                    # the recovery was not instantaneous
                                    not_insta.append(seed)
                                    not_insta_location.append((seed,p))
                                    break

                        # recovery happened at (tour, kappa) where projection was not found
                        else:
                            not_proj_then_e.append(seed)

                            # check earlier (all in this case) projections
                            for p in found_proj:
                                # pi(e) found at an early kappa, and in a position where size_reduction will consider it, so <= kappa
                                if p[1] < cutoff and p[2] <= p[1]:
                                    # the recovery was not instantaneous
                                    not_insta.append(seed)
                                    not_insta_location.append((seed,p))
                                    break

            # error vector was not found
            else:
                # add to list of non recoveries
                not_recover.append(seed)

                # get the info about found projections
                found_proj = inst['proj'] # = [(tour_n, kappa, e_index),...]

                # for every projection found, check if it was not too late
                for p in found_proj:
                    # pi(e) found at an early kappa, and in a position where size_reduction will consider it, so <= kappa
                    if p[1] < cutoff and p[2] <= p[1]:
                        not_rec_but_early_proj.append(seed)
                        break

        # experiment did not terminate
        else:
            not_terminated.append(seed)

    # normalise pmf for where we recover e
    pmf_k_where_e_is_recovered = [x/len(recover) for x in pmf_k_where_e_is_recovered]
    pmf_mean = 0
    for i in range(len(pmf_k_where_e_is_recovered)):
        pmf_mean += (i+1)*RR(pmf_k_where_e_is_recovered[i])
    pmf_tikz = ""
    for i in range(len(pmf_k_where_e_is_recovered)):
        pmf_tikz += "(%3d, %.4f) "%(i+1, pmf_k_where_e_is_recovered[i])


    # make a nice plot
    plot = list_plot(pmf_k_where_e_is_recovered, color="blue", plotjoined=True)
    plot += line([(dim-bs, 0), (dim-bs, 0.15)], color="black", thickness=0.5, legend_label=r"$d-\beta+1$")
    plot += line([(dim-0.9*bs, 0), (dim-0.9*bs, 0.15)], color="red", thickness=0.5, legend_label=r"$d-0.9\beta+1$")
    plot.set_legend_options(loc=1)
    # save(plot, "experiments/b%dd%d-%s.png"%(bs,dim,label))

    print "Blocksize %d dimension %d"%(bs,dim)
    print "Total number of experiments: %d"%total
    print "Total cpu time in hrs: %f"%(cputime/60/60)
    print "Experiments successfully run (no fpylll bug): %d"%len(terminated)
    print
    print "--Successful recovery--"
    print "Experiments where e was recovered:                       %d/%d"%(len(recover),len(terminated))
    print "Experiments where e was recovered from a late kappa:     %d"%len(recovered_late)
    print "Experiments where e is recovered timely:                 %d (= %d)"%(len(recovered_early),len(recover) - len(recovered_late))
    print "e not recovered by the last size reduction:              %d/%d"%(len(not_last_size_reduction),len(recovered_early))
    print "proj(e) was found twice before recovering e (sr failed): %d/%d"%(len(not_insta),len(recovered_early))
    print "e was recovered without first seeing any proj(e):        %d (should be 0)"%len(not_proj_found_but_recover)
    print "e was recovered without seeing proj(e) ~in the same (tour, kappa)~ (excluded above): %d (this should be 0)"%len(not_proj_then_e)
    print
    print "--No error vector recovered--"
    print "proj(e) was found but e was not recovered:               %d (this should be 0, except if proj(e) got stuck at the last tour)"%len(not_rec_but_early_proj)
    print
    print "--SR failure detail--"
    print "When was it found twice:"
    print not_insta_location
    print

    return {"cputime": cputime,
            "recover": recover,
            "not_recover": not_recover,
            "terminated": terminated,
            "not_terminated": not_terminated,
            "not_last_size_reduction": not_last_size_reduction,
            "recovered_late": recovered_late,
            "found_many_e": found_many_e,
            "not_proj_then_e": not_proj_then_e,
            "not_proj_found_but_recover": not_proj_found_but_recover,
            "not_insta": not_insta,
            "not_rec_but_early_proj": not_rec_but_early_proj,
            "pmf_k_where_e_is_recovered": pmf_k_where_e_is_recovered,
            "pmf_mean": pmf_mean,
            "pmf_tikz": pmf_tikz,
            "plot_pmf_k_where_e_is_recovered": plot,
            "not_insta_location": not_insta_location,
            "plot": plot}


def sameStepRecoveryRatio(returns,
                          pos):
    """Counts the proportion of instances where the short vector was recovered
       in the same step as when the projection is found (i.e. it's the size
       reduction after the SVP oracle that recovers the full vector), for a given
       basis indexs.

    :param returns:     Parallel batch output
    :param pos:         Basis index where we are looking at, e.g. b-d+1.

    :returns:           Object containing the results.
    """
    proj_at_pos = []
    fail_at_pos = []

    # start reading the records
    for x in returns:
        seed = x[0][0][0]
        inst = x[1]

        # if experiment terminated
        if 'vec' in inst:

            # check if a projection with that step was ever found
            # then discern wether it lead to v or not


            # get info about found projections
            proj = inst['proj'] # = [(tour_n, kappa, e_index), ...]

            # for all projs
            for i in range(len(proj)):
                p = proj[i]
                # if found a projection at the correct position
                if p[2] == pos:
                    proj_at_pos.append(seed)
                    if i+1 < len(proj):
                        # you failed to recover at that step
                        fail_at_pos.append(seed)
                    break

    print "--Checking recovery of v from \pi_pos(v)--"
    print "Pos: %d"%pos
    print "Proj at pos: %d"%len(proj_at_pos)
    print "Fail at pos: %d"%len(fail_at_pos)
    if len(proj_at_pos) > 0:
        print "Fail ratio:  %3.4f pc"%(RR(100.0*len(fail_at_pos)/len(proj_at_pos)))
        print "Same step ratio: %.4f pc"%(RR(100.0-(100.0*len(fail_at_pos)/len(proj_at_pos))))
    print
    return {
        "proj_at_pos": proj_at_pos,
        "fail_at_pos": fail_at_pos
    }


def gen_plot(n,
             q,
             sd,
             m,
             bs,
             stats,
             output_dir="."):
    """Extract lengths from statistic object to generate plots

    :param n:               secret dimension
    :param q:               lwe modulo
    :param sd:              standard deviation
    :param m:               number of lwe samples
    :param bs:              bkz block size
    :param stats:           statistics object
    :param output_dir:      directory where plots will be saved

    """


    # ideal predictions
    gsa_norms, id_estar = gsa_prediction(n, q, sd, m, bs)

    # analyse results
    counter = 0
    print "plots", len("%s"%stats["plots"])
    for tour in stats["plots"]:
        for step in stats["plots"][tour]:
            gso_norms = stats["plots"][tour][step]["gso_norms"]
            # ||v_i^*|| norms are by default not recorded
            # Look into StatBKZ.TrackSuccess to enable them
            # evc_norms = stats["plots"][tour][step]["evc_norms"]
            evc_norms = None
            gen_step_plot(m+1,
                          bs,
                          output_dir,
                          step,
                          "%05d"%counter,
                          gsa_norms=gsa_norms,
                          id_estar=id_estar,
                          gso_norms=gso_norms,
                          evc_norms=evc_norms)
            counter += 1


def gsa_prediction(n,
                   q,
                   sd,
                   m,
                   bs):
    """Generate GSA prediction for vector lengths and usvp projection lengths.
    It assumes Kannan embedding, and Discrete Gaussian error vector

    :param n:               secret dimension
    :param q:               lwe modulo
    :param sd:              standard deviation
    :param m:               number of lwe samples
    :param bs:               bkz block size

    :returns:
    """

    # Kannan embedding
    vol = q**(m-n)
    d = m+1

    # [Chen13] asymptotic for Hermite factor
    delta_0 = (bs/(2*pi*e) * (pi*bs)**(1./bs))**(1./(2*bs-1))

    # [ADPS16] GSA + Hermite factor => GSA slope
    alpha = delta_0**(-2*n/(n-1))

    # Hermite factor definition
    b_1 = (delta_0**d)*((vol)**(1/d))

    # ''ideal'' lengths
    gsa_norms = map(lambda x: RR(log(x, 2)), [(alpha**i * b_1)**2 for i in range(d)])
    id_estar = map(lambda x: RR(log(x, 2)), [(sd*sqrt(d-i))**2 for i in range(d)])

    return gsa_norms, id_estar


def gen_step_plot(dim,
                  bs,
                  output_dir,
                  step,
                  fn,
                  gsa_norms=None,
                  id_estar=None,
                  gso_norms=None,
                  evc_norms=None):
    """Generate a log plot for the basis lengths

    :param dim:             embedding lattice dimension
    :param bs:              BKZ block size
    :param output_dir:      plot output directory
    :param step:            tour's step, iterating from 0 to d-1
    :param fn:              filename
    :param gsa_norms:       GSA expected lengths
    :param id_estar:        expected usvp projections' lengths
    :param gso_norms:       actual basis vector norms
    :param evc_norms:       actual short vector projection norms

    :returns:               tikzpicture code
    """

    base = "#5DA5DA"
    colour = "#FAA43A"
    ymax = ceil(max(gsa_norms+(gso_norms or [])))+1

    # tikz header
    tikz = """\\begin{tikzpicture}
    \\begin{axis}[
    /pgf/number format/.cd,
    width = .9\columnwidth,
    height = 0.5\columnwidth,
    fixed,
    grid = both,
    xlabel = %s,
    ylabel = %s,
    xmin = %s,
    xmax = %s,
    ymin = 0,
    ymax = %s,
    legend cell align=left,
    legend pos = north east,
    try min ticks = 7,
    ]
    """%("basis index $i$",
         "$\,\log_2\,(\,\\text{lengths}\,)$",
         0,
         dim-1,
         ymax)


    def _tikzline(points, width, colour, legend):
        s = "\n\\addplot[line width=%s, %s] coordinates {\n"%(width, colour)
        for p in points:
            s += " (%d, %.5f) "%(p[0], p[1])
        s += "\n};\n\\addlegendentry{%s}\n"%legend
        return s

    # current block being reduced
    g = polygon([
                 (step, 0),
                 (step, ymax),
                 (min(step+bs-1,dim-1), ymax),
                 (min(step+bs-1,dim-1), 0)
                ],
                color="#000000",
                alpha=0.2,
                legend_label="current block")

    tikz += "\\fill [black, opacity=0.2] (%s,0) rectangle (%s,%s);"%(
        step*10,
        min(step+bs-1,dim-1)*10,
        ymax*10
    )

    # GSA line
    if gsa_norms:
        g += line(zip(range(dim), gsa_norms),
                  legend_label="GSA",
                  color=base,
                  frame=True,
                  axes=False,
                  transparent=True,
                  axes_labels=["index $i$", "$2\\,\\log_2 \\|\mathbf{b}^*_i\\|$"])

        tikz += _tikzline(zip(range(dim), gsa_norms), "3pt", "orange", "GSA")

    # measured GS basis norms"
    if gso_norms:
        g += line(zip(range(dim),gso_norms),
                  legend_label="actual GS norms",
                  color=colour)

        tikz += _tikzline(zip(range(dim), gso_norms), "3pt", "red", "actual GS norms")

    # measured projections of short vector
    if evc_norms:
        g += line(zip(range(dim),evc_norms),
                  legend_label="actual $\\|\\mathbf{e}^*_{i}\\|$",
                  color=colour,
                  alpha=0.5,
                  linestyle='--')

        tikz += _tikzline(zip(range(dim), evc_norms), "3pt", "green", "actual $\\|\\mathbf{e}^*_{i}\\|$")

    # Predicted projections of the short vector
    if id_estar:
        g += line(zip(range(dim), id_estar),
                  legend_label="$\\|\\mathbf{e}^*_{i}\\|$",
                  color=base,
                  alpha=0.5,
                  linestyle='--')

        tikz += _tikzline(zip(range(dim), id_estar), "3pt", "teal", "$\\|\\mathbf{e}^*_{i}\\|$")

    tikz += """\end{axis}
    \end{tikzpicture}
    """

    # save plot
    save(plot(g),
         "%s/%s.png"%(output_dir,fn),
         dpi=300,
         ymin=0,
         ymax=ymax
        )

    with open("%s/%s.tex"%(output_dir,fn), "w") as f:
        f.write(tikz)


def example():
    """A short example for reproducing the results or running further
    experiments.
    """

    # (Re)running experiments
    batch = experiment_parameters["n65"]
    ret = list(parallelExperiment(batch))
    # save result (if save procedure not defined in batch parameters)
    if not batch[0][9]:
        save(ret, "experiment_results.sobj")
    # Obtain satistics from the experiments
    extractExperimentResults(ret, batch[0][5], batch[0][4]+1)
    sameStepRecoveryRatio(ret, batch[0][4]+1-batch[0][5]+1) # (m+1)-b+1


    # # Load precomputed saved results
    # ret = load("experiments_results.sobj")
    # Obtain satistics from loaded results
    # extractExperimentResults(ret, b, m+1, "some label")
    # sameStepRecoveryRatio(ret, m+1-d+1)


    # # Run experiment to generate plots
    # exp = experiment_parameters["n65"][0]
    # exp[-1] = True # set plots=True
    # exp[8] = 1 ##################################
    # ret = list(parallelExperiment([exp]))
    # gen_plot(exp[1], exp[2], exp[3], exp[4], exp[5], ret[0][1], output_dir="plots")
