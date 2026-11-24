"""The one place a user edits to change what the feature-selection pipeline does.

Every knob the pipeline consults lives here: the hand-picked feature lists, the
hand-defined clusters, how many features each selector is asked for, and the full
hyperparameter search space of every clustering algorithm. Nothing downstream should
hardcode a threshold, a grid or a feature name -- if a search parameter is not in this
file, it is hidden, and a reader cannot tell what was searched from what was assumed.

Two constants are deliberately *not* redefined here:

* ``RANDOM_STATE`` is re-exported from ``allmeta.paths``. A second seed constant is
  exactly the drift ``columns.py``'s docstring was written to prevent -- two files
  disagreeing about the seed would make two analyses silently incomparable.
* the descriptor vocabulary stays in ``allmeta.columns``. This module names *subsets*
  of it; it does not redeclare it.

Call ``validate()`` before using anything here. It is cheap, and every failure it
catches (a typo'd descriptor, a feature in two clusters, the target smuggled into the
feature list) otherwise surfaces hundreds of lines later as a confusing KeyError or,
worse, as a silently wrong number.
"""

from allmeta import columns as cols
from allmeta.paths import RANDOM_STATE  # noqa: F401  -- re-exported on purpose

# ---------------------------------------------------------------------------
# 1. Manual feature subset  (implementation plan, Step 2.1)
# ---------------------------------------------------------------------------

# The hand-picked shortlist. Seeded from columns.MANUAL_KEEP, which was carried over
# verbatim from notebooks/reduced_cleaning.ipynb. Paste your own list here; order is
# preserved through the pipeline, so a meaningful ordering survives into the outputs.
MANUAL_FEATURES: list[str] = list(cols.MANUAL_KEEP)


# ---------------------------------------------------------------------------
# 2. Manual feature clusters  (implementation plan, Step 2.4)
# ---------------------------------------------------------------------------

# PLACEHOLDER -- a domain-informed starting point, not a settled chemistry claim.
#
# Built by reading the measured feature correlation structure on subset B and then
# grouping by MECHANISM rather than by the dendrogram, because the manual set is the
# control the automatic clustering gets judged against. A grouping that merely
# reproduced the dendrogram would have nothing to say when the algorithms reproduce it
# too.
#
# The 13 composition/atom-count columns (columns.COMP_COLUMNS) are excluded, matching
# columns.screening_features(). The remaining 45 descriptors are partitioned exactly.
#
# Three descriptors are deliberately held out of the volatility family they correlate
# with, because for air-displacement dispensing they act through different mechanisms
# and the point of the manual set is to let the analysis test that:
#
#   log eta   correlates 0.93 with Ant1T and 0.91 with dHv@BPt, but viscosity acts on
#             tip drainage and film retention, not on evaporation. It is also
#             columns.PROTECTED. Keeping it separate is what makes "does viscosity add
#             anything beyond the volatility axis?" an answerable question.
#   SurfTen   correlates 0.78 with Tc, but drives tip wetting and droplet retention.
#   MPt       correlates 0.61-0.66 with the family. Melting behaviour is not relevant
#             to evaporation from a tip at 25 C.
#
# Two more are singletons because the data says they belong nowhere: Zc correlates at
# most 0.42 with anything, LogOHR at most 0.49.
#
# Measured PC1 explained variance per multi-member cluster on subset B (720 liquids,
# the 12 benchmarked ones excluded): partial_charge 1.00, formation_thermo 0.97,
# hbond_donor 0.95, molecular_size 0.83, hydrophobicity 0.75, volatility 0.73,
# dispersion 0.72, vaporisation_shape 0.67, hbond_acceptor 0.52.
#
# hbond_acceptor is the weak one: dH/dHAcc are tight with each other but Henry, Abr-b
# and Cond. only join them at |r| ~ 0.65-0.70. It is kept as one family because a
# chemist would call it one, and it is the clearest candidate for an automatic method
# to improve on.
#
# Rules the validator enforces: no feature in two clusters, no empty cluster, every
# name present in columns.NUM_COLUMNS. Singleton clusters are legal -- they pass
# through as a standardised raw feature rather than a PC1.
MANUAL_CLUSTERS: dict[str, list[str]] = {
    # Where the liquid boils and how hard it is to evaporate. The dominant family in
    # this dataset and, given that most benchmarked liquids are volatile, expected to
    # carry most of the predictive signal.
    'volatility': ['BPt', 'Tc', 'AntB', 'AntC', 'Ant1T', 'ΔHv@BPt', 'RER'],
    # The SHAPE of the vapour-pressure curve rather than its position: Trouton's
    # constant is dHv/Tb, AntA is the Antoine intercept, EdmiW an acentric-factor
    # relative. Mutually correlated at 0.59-0.73 and only weakly tied to BPt.
    'vaporisation_shape': ['AntA', 'Trouton', 'EdmiW'],
    # Everything that scales with how big the molecule is. Pc is included despite
    # being inversely related: 1 - |r| is sign-blind and PCA handles the sign.
    'molecular_size': ['MWt', 'MVol', 'Area', 'Ovality', 'Pc', 'Vc', 'Parachor', 'RD',
                       'MCI', 'Sym', 'ΔHfus', 'ΔHcomb', 'ΔCp'],
    'hbond_donor': ['δHDon', 'Abr-a'],
    'hbond_acceptor': ['δH', 'δHAcc', 'Henry', 'Abr-b', 'Cond.'],
    'hydrophobicity': ['LogKow', 'LogS', 'δP'],
    # Polarisability and how tightly the liquid packs.
    'dispersion': ['δD', 'RI', 'Density'],
    'partial_charge': ['MaxPc', 'MinMc'],
    'formation_thermo': ['ΔHform', 'ΔGform'],
    # Held out of volatility on mechanism -- see the note above.
    'surface': ['SurfTen'],
    'viscosity': ['log η'],
    'melting': ['MPt'],
    # Held out because nothing correlates with them.
    'compressibility': ['Zc'],
    'oxidation': ['LogOHR'],
}


# ---------------------------------------------------------------------------
# 3. mRMR and direct PCA  (Steps 2.2, 2.3)
# ---------------------------------------------------------------------------

# How many features mRMR is asked for. Deliberately small: with 13 labelled liquids a
# training fold holds 12, so a larger request is fitting the fold rather than the
# chemistry. Kept in step with allmeta.supervised.N_SELECT so the new pipeline and the
# existing leave-one-liquid-out selection ask for the same thing.
MRMR_N_FEATURES: int = 5

# feature_engine variant. MID/MIQ estimate both relevance and redundancy by mutual
# information; FCD/FCQ use an F-statistic against correlation; RFCQ uses random-forest
# importance. The Q (ratio) variants divide relevance by redundancy, so a zero-variance
# column scores maximally -- the constant guard in mrmr.select_mrmr exists for that.
MRMR_METHOD: str = 'FCD'

# Components retained by the direct PCA of the full raw feature set. A float in (0, 1)
# is read by sklearn as a cumulative-explained-variance target rather than a count.
FULL_PCA_N_COMPONENTS: float | int = 0.95


# ---------------------------------------------------------------------------
# 4. Where the unsupervised transforms are fitted
# ---------------------------------------------------------------------------

# Feature clustering and PCA are fitted on an unlabelled subset and *applied* to the
# labelled liquids, rather than refitted inside each outer fold.
#
# The reason is sample size. Subset C holds 13 liquids; an outer training fold holds
# 12 and an inner one about 11. Choosing among hundreds of clustering configurations
# against 11 liquids selects noise, and the one-standard-error rule does not repair
# that -- it widens the tie-break, it does not add information. Correlation structure
# between features, on the other hand, is estimated on 1206 liquids (A) or 732 (B),
# and those subsets carry no target at all, so nothing fitted on them can leak the
# label. The 13 labelled liquids are spent only on held-out evaluation.
#
# 'A' = every liquid with a complete descriptor row; 'B' = A restricted to the curated
# liquid list. B is the default: the curation removed liquids on chemical grounds, and
# a correlation family that exists only among the removed ones is not one we want to
# build a factor from.
CLUSTER_FIT_SUBSET: str = 'B'

# The 13 labelled liquids are *members* of subsets A and B, so a transform fitted on
# all of B has seen the held-out liquid's descriptors -- not its label, but its row.
# With this True the outer-test liquid is dropped from the fit. It costs one row in
# 732 and removes the objection outright, which is cheaper than arguing about whether
# transductive access to an unlabelled row matters.
EXCLUDE_LABELLED_FROM_CLUSTER_FIT: bool = True


# ---------------------------------------------------------------------------
# 5. Clustering algorithms and their search spaces  (Phases 4, 8, 9)
# ---------------------------------------------------------------------------

# The algorithms are implemented in a later pass; the grids live here from the start so
# that no search parameter is ever buried in algorithm code.
CLUSTERING_METHODS: list[str] = [
    'hierarchical',
    'hdbscan',
    'kmeans',
    'kmedoids',
    'gmm',
]

# 'sequential' tunes each algorithm on its own and then compares the winners;
# 'joint' searches algorithm and hyperparameters as one space.
HYPERPARAMETER_OPTIMIZATION_MODE: str = 'sequential'

# 'winner' carries one optimised clustering solution forward; 'retain_competitive'
# carries every statistically competitive method forward as its own feature set.
CLUSTERING_COMPARISON_MODE: str = 'winner'

# What to do with raw manual/mRMR features that are still redundant after Elastic Net:
# 'none' | 'representative' (one member per cluster) | 'cluster_pc1'.
#
# 'none' is the default and, on this data, almost certainly the right one. Phases 10-12
# measured one standard error wider than the entire gap between the best model and the
# mean baseline, so a further reduction chosen on the same 13 liquids would be chosen on
# noise. The clusters are still reported under 'none' -- the diagnostic is worth having
# even when nothing is done with it.
FINAL_RECLUSTER_MODE: str = 'none'

# Correlation distance at which surviving raw features are grouped in Phase 13.
# 0.15 means "group while members correlate above |r| = 0.85". Chosen to match the cut
# the Phase 4 dendrogram showed the volatility family separating at, so the post-Elastic
# Net grouping and the initial one are answering the same question at the same
# resolution rather than at two arbitrary thresholds.
POST_ELASTIC_NET_DISTANCE_THRESHOLD: float = 0.15

# Whether Phase 14 re-runs the Phase 12 stability-threshold search inside every outer
# fold. False by default and the cost is the reason: the threshold search refits Elastic
# Net across every resample of every inner fold, and nesting that inside 13 outer folds
# multiplies the run by roughly an order of magnitude.
#
# This is a real methodological compromise, not a free one. With it False, Phase 14
# evaluates the pipeline "tune Elastic Net, keep its non-zero coefficients" rather than
# "tune Elastic Net, then threshold on selection frequency". The thresholded variant is
# measured honestly by elastic_net.threshold_search's nested scope, which is nested
# within its own folds; it is simply not carried into the outer-fold comparison. Set
# True to include it and expect the run to take hours rather than minutes.
NESTED_INCLUDE_THRESHOLD_SEARCH: bool = False

# Which representation each clustering algorithm consumes. Declared here rather than
# inside the algorithms so that "what does this method actually see?" is answerable
# without reading five modules, and so validate() can reject a method with no
# representation. The representations themselves are built in representations.py.
REPRESENTATION_BY_METHOD: dict[str, str] = {
    'hierarchical': 'correlation_distance',
    'hdbscan': 'correlation_distance',
    'kmedoids': 'correlation_distance',
    'kmeans': 'feature_profile',
    'gmm': 'feature_profile',
}


# Consumes the correlation-distance representation, d = 1 - |r|. Ward linkage is
# absent on purpose: it minimises Euclidean inertia and is undefined for a precomputed
# non-Euclidean distance. Thresholds are on that distance, so 0.10 means "merge while
# members correlate above |r| = 0.90".
HIERARCHICAL_CONFIG: dict = {
    'correlation_methods': ['pearson', 'spearman'],
    'linkages': ['average', 'complete', 'weighted'],
    'distance_thresholds': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
}

# sklearn.cluster.HDBSCAN, native since scikit-learn 1.3 -- no extra dependency.
# min_cluster_size is in *features*, not liquids, so 2 is the smallest meaningful
# value on a 45-descriptor problem.
HDBSCAN_CONFIG: dict = {
    'min_cluster_size': [2, 3, 4, 5],
    'min_samples': [None, 1, 2, 3],
    'cluster_selection_epsilon': [0.0, 0.05, 0.10],
    'cluster_selection_method': ['eom', 'leaf'],
    'metrics': ['precomputed'],
}

# HDBSCAN labels an unassigned object -1. Dropping those features would silently
# shrink the feature set, so a noise feature becomes its own singleton cluster and
# survives as a standardised raw feature. 'singleton' | 'own_cluster' | 'drop'.
HDBSCAN_NOISE_POLICY: str = 'singleton'

# Consumes the Euclidean feature-profile representation (features as rows). n_clusters
# must stay below the feature count; the search filters that at run time.
KMEANS_CONFIG: dict = {
    'n_clusters': list(range(3, 16)),
    'n_init': [10],
}

# Consumes the correlation-distance matrix directly (metric='precomputed'), which is
# the reason to prefer medoids over means here: the medoid of a correlation family is
# an actual descriptor, and no Euclidean embedding has to be invented first.
KMEDOIDS_CONFIG: dict = {
    'n_clusters': list(range(3, 16)),
    'init': ['random', 'first', 'build'],
    'metric': ['precomputed'],
}

# Consumes the Euclidean feature-profile representation.
#
# MEASURED WARNING, kept here rather than in a report because it decides whether these
# numbers should be believed. The profile space is 45 descriptors in 44 dimensions, and
# every covariance type in this grid is over-parameterised at that shape:
#
#     spherical   ~405 free parameters      diag  ~792      tied  ~1386
#
# all fitted from 45 points. The symptom is unmistakable once looked for: every fit
# reports mean posterior assignment confidence of exactly 1.0 and converges in 2-8 EM
# iterations. That is not a mixture that found components, it is one that has collapsed
# onto its points; reg_covar prevents an outright singular matrix but not the
# degeneracy. 'full' is excluded, but excluding it was not enough.
#
# The rest of the evidence agrees: GMM has the lowest correlation separation of the
# five methods (median delta 0.28), the lowest PC1 retention (0.649 over multi-member
# clusters, 43% of them below the retention floor), and produces the single worst
# cluster in the entire sweep -- 21 descriptors spanning Abr-b through Zc, retaining
# 0.205. 'tied' is also 22x slower per fit than the others and dominates the stability
# sweep's cost.
#
# It is kept because the plan requires GMM to be implemented and compared, and because
# a method should be removed on a stated criterion in Phase 8 rather than on a hunch
# here. Treat its rows as a control, not a candidate.
GMM_CONFIG: dict = {
    'n_components': list(range(3, 16)),
    'covariance_type': ['diag', 'spherical', 'tied'],
    'reg_covar': [1e-6, 1e-4, 1e-2],
    'n_init': [5],
}


# ---------------------------------------------------------------------------
# 5b. Clustering stability  (Phase 7)
# ---------------------------------------------------------------------------

# How many subsamples of the fitting frame each clustering candidate is refitted on.
# Stability is a first-class criterion here, not a footnote: with 13 labelled liquids
# downstream, a partition that reshuffles when the liquid set changes cannot support a
# claim about anything.
#
# 20 is chosen from measured cost, not convention. Two quantities grow differently:
# refitting the grid is linear in B and costs ~35 s per resample, while pairwise
# agreement is quadratic at B(B-1)/2 comparisons per candidate and costs ~0.19 ms each.
# Measured on the real grid:
#
#     B=10   45 pairs/candidate    ~6 min
#     B=20  190 pairs/candidate   ~12 min     <- the default
#     B=30  435 pairs/candidate   ~18 min
#     B=50 1225 pairs/candidate   ~34 min
#
# At B=20 the mean pairwise ARI is precise enough to rank methods that differ by more
# than about 0.05, and a consensus probability has a standard error near 0.11. B=50
# would take it to 0.07 for nearly three times the time, which changes no conclusion
# this analysis draws. Below B=10 the mean ARI is too noisy to rank anything.
#
# Almost all of the refitting cost is GMM: 39 of the 430 candidates -- GMM with
# covariance_type='tied' -- account for roughly three quarters of it at 665 ms a fit
# against 0.3-2 ms for the distance-based methods. See GMM_CONFIG.
STABILITY_N_RESAMPLES: int = 20

# Fraction of liquids in each subsample. 0.8 matches the convention already used by
# allmeta.clustering.RESAMPLE_FRACTION, so the two stability numbers in this repository
# mean comparable things even though one resamples to cluster liquids and the other to
# cluster features.
STABILITY_SUBSAMPLE_FRACTION: float = 0.8

# Cap on candidates per algorithm, evenly spaced through each grid. None runs the whole
# grid. This is the lever that makes a smoke test cheap without changing the method.
STABILITY_MAX_CANDIDATES_PER_METHOD: int | None = None

# Cap on liquids in the frame before resampling. None uses every liquid. Reducing it is
# not only a speed knob: rerunning with 200 against 720 answers "is this partition
# stable because the structure is real, or because 720 liquids would stabilise
# anything?", which is worth knowing on a project this sample-limited.
STABILITY_FRAME_SIZE: int | None = None

# Named presets, so a smoke test is one argument rather than four.
#   'full'  -- 20 resamples, whole grid, every liquid. ~12 minutes. The one to quote.
#   'quick' -- 10 resamples, whole grid, every liquid. ~6 minutes. Same method, half the
#              precision: 45 pairs per candidate instead of 190. Fine for seeing which
#              way the ranking points before committing to a full run.
#   'test'  -- 5 resamples, 6 candidates per algorithm, 200 liquids. ~7 seconds. Enough
#              to prove the machinery runs and NOT enough to rank anything: 10 pairs
#              per candidate gives a mean ARI with no useful precision. Results carry
#              their settings so this cannot be mistaken for a full run.
STABILITY_PRESETS: dict[str, dict] = {
    'full': {
        'n_resamples': STABILITY_N_RESAMPLES,
        'subsample_fraction': STABILITY_SUBSAMPLE_FRACTION,
        'max_candidates_per_method': STABILITY_MAX_CANDIDATES_PER_METHOD,
        'frame_size': STABILITY_FRAME_SIZE,
    },
    'quick': {
        'n_resamples': 10,
        'subsample_fraction': STABILITY_SUBSAMPLE_FRACTION,
        'max_candidates_per_method': None,
        'frame_size': None,
    },
    'test': {
        'n_resamples': 5,
        'subsample_fraction': 0.8,
        'max_candidates_per_method': 6,
        'frame_size': 200,
    },
}


# A cluster whose PC1 retains less than this is flagged in
# clustering_pc1_quality.csv. Not a filter and not a cut -- nothing is dropped for
# failing it. It is the line below which "the factor summarises the cluster" stops
# being true: at 0.60 the discarded components still hold more than a third of what the
# cluster carried, so the factor is a lossy stand-in rather than a summary of it.
PC1_RETENTION_FLOOR: float = 0.60


# ---------------------------------------------------------------------------
# 5c. Selecting a clustering candidate  (Phase 8)
# ---------------------------------------------------------------------------

# The common downstream model every candidate is scored through. Ridge, at a FIXED
# alpha, deliberately: tuning alpha per candidate would give each one its own best case
# and turn a controlled comparison into a contest of who overfits 11 liquids best. The
# question here is which feature representation helps, holding the model constant.
BASELINE_RIDGE_ALPHA: float = 1.0

# Which loss drives the selection. Both are computed and written either way.
#
#   'liquid_mse'   squared error of each held-out liquid's MEAN prediction
#   'pooled_rmse'  per-fold RMSE over all 104 rows
#
# 'liquid_mse' is the default because pooled RMSE cannot discriminate on this data.
# A liquid's eight volumes share one descriptor vector, so 56.5% of the target's
# variance is within-liquid and no liquid-level feature can touch it. That unreachable
# part enters every candidate's pooled error identically, as noise: measured on the real
# grid, one standard error came to 17% of the entire spread across 430 candidates and
# the one-SE band swallowed 74 of them, at which point the tie-break rather than
# prediction was choosing. Scoring the same models on the liquid means instead drops the
# constant and narrows the band to 13, every one of which also beats the mean baseline.
#
# This is not a departure from the plan's "predictive RMSE is primary" -- it is that
# rule measured where this target has signal. Set 'pooled_rmse' to get the literal
# reading; the columns for both are in every output.
SELECTION_CRITERION: str = 'liquid_mse'

# Refuse to select a candidate that loses to predicting the training mean. The one-SE
# rule chooses among candidates indistinguishable from the BEST one, which says nothing
# about whether the best is any good; without this guard a grid where everything fails
# still returns a confident-looking winner.
REQUIRE_BEATS_BASELINE: bool = True

# Order in which the one-standard-error tie-break is applied among candidates that are
# statistically indistinguishable on predictive error. Straight from the plan's Step
# 8.3, with the direction each is read in.
SELECTION_TIE_BREAK: tuple[tuple[str, str], ...] = (
    ('n_factors', 'lower'),                            # parsimony first
    ('ari_mean', 'higher'),                            # then stability
    ('within_between_delta', 'higher'),                # then correlation coherence
    ('mean_pc1_explained_variance_multi', 'higher'),   # then PC1 retention
    ('silhouette_correlation', 'higher'),              # then internal geometry
)

# The plan's literal Step 8.3 ordering, kept runnable under its own name so the
# comparison in TIE_BREAK_ORDERINGS is against something real rather than a description
# of what the code used to do.
SELECTION_TIE_BREAK_PLAN_LITERAL: tuple[tuple[str, str], ...] = SELECTION_TIE_BREAK

# Orderings that comparison.replay_tie_breaks re-applies to already-scored search
# tables. No model is refitted, so this costs nothing and every ordering sees identical
# candidates.
#
# WHY THIS EXISTS. On the diagnostic search the global one-SE band holds 13 candidates
# whose held-out between-liquid R2 runs from +0.106 to -0.140. The rule's premise is
# that everything inside the band is indistinguishable, so the tie-break may choose on
# other grounds at no cost in error; a band that wide at n=13 means the premise is a
# statement about the sample size rather than about the candidates. Measured on the
# diagnostic search, the best candidate on BOTH the criterion and held-out R2 (kmedoids,
# 12 factors, +0.106) is selected by NO ordering built from these columns:
#
#     parsimony_first  -> gmm, 3 factors       (R2 -0.001)
#     retention_first  -> kmedoids, 13 factors (R2 -0.033)
#     criterion_only   -> kmedoids, 12 factors (R2 +0.106)  -- i.e. ignore the band
#
# So "promote retention above parsimony" does not recover it either, and on that single
# draw retention_first is WORSE on held-out R2 than the plan's own ordering. Whether
# that survives across the 13 outer folds is what the replay measures.
TIE_BREAK_ORDERINGS: dict[str, tuple[tuple[str, str], ...]] = {
    'parsimony_first': SELECTION_TIE_BREAK_PLAN_LITERAL,
    'retention_first': (
        ('mean_pc1_explained_variance_multi', 'higher'),
        ('ari_mean', 'higher'),
        ('within_between_delta', 'higher'),
        ('n_factors', 'lower'),
        ('silhouette_correlation', 'higher'),
    ),
    'stability_first': (
        ('ari_mean', 'higher'),
        ('mean_pc1_explained_variance_multi', 'higher'),
        ('n_factors', 'lower'),
        ('within_between_delta', 'higher'),
        ('silhouette_correlation', 'higher'),
    ),
    # No tie-break: take the outright minimum. This is what the one-SE rule exists to
    # AVOID, and it is included as the control that says how much the rule costs here.
    'criterion_only': (),
}


# ---------------------------------------------------------------------------
# 5d. Which clustering methods are candidates rather than controls  (Phase 9)
# ---------------------------------------------------------------------------

# A method must clear this gate to be carried forward into Elastic Net. Every criterion
# reads a FITTED-MODEL DIAGNOSTIC and none of them can see the target.
#
# That restriction is the entire design. GMM wins the diagnostic search and 10 of the 13
# per-fold selections while holding the worst correlation separation, the worst PC1
# retention, the worst stability and a diagnosable degeneracy. Two readings of that are
# available -- cluster quality does not predict usefulness, or the predictive signal is
# so weak the winner is noise -- and with 16 of 430 candidates beating the baseline at a
# 1.13-SE margin, the evidence cannot separate them. A gate built from predictive error
# would be a gate built from the outcome we would prefer, which is not a criterion. This
# one cannot see the outcome at all.
#
# A method that fails keeps every row it has in every metric, stability and search
# table. It is dropped from carry_forward and from nothing else.
CLUSTERING_METHOD_ADMISSION: dict = {
    # A mixture reporting a maximum posterior of exactly 1.00 for every one of 45
    # descriptors, converging in 2-8 EM iterations, has collapsed onto its points.
    # reg_covar prevents a singular covariance, not the degeneracy. Methods that produce
    # no posterior leave this NA and pass vacuously -- an absent diagnostic is not a
    # failed one.
    'max_mean_assignment_confidence': 0.999,
    # Reuses PC1_RETENTION_FLOOR, set in Phase 6 before any predictive number existed.
    # Below it the factor is not summarising its cluster, it is discarding it.
    'min_pc1_explained_variance_multi': PC1_RETENTION_FLOOR,
    # A partition that does not survive resampling cannot support a claim about which
    # descriptors belong together.
    'min_ari_mean': 0.70,
    'require_all_converged': True,
}

# Whether the gate judges a method by the median of its WHOLE GRID ('grid') or by the
# single candidate the tie-break selected ('winner').
#
# 'grid' is the default and the correct level. Evaluated on the winner, the gate rejects
# GMM by 0.000514 -- retention 0.599486 against a 0.60 floor -- and that verdict is
# contingent on the tie-break, because GMM's winner is the 3-component solution only
# under parsimony-first, and the 6-component solution retains 0.6238 and would pass. The
# outer-fold replay shows the tie-break ordering is itself indistinguishable from noise
# (paired Wilcoxon p = 0.11-0.75 across 13 folds), so a winner-scoped gate hands its
# answer to a coin flip.
#
# At grid scope there is no such dependence and no marginal call: GMM fails two
# independent criteria (median ARI 0.679 against 0.70, median mean-posterior exactly
# 1.0000 against 0.999) and every other method passes every criterion. The degeneracy
# was always a claim about GMM at this data shape, which is a property of the grid.
ADMISSION_SCOPE: str = 'grid'

# The plan's final tie-break is "greater interpretability", which is not a column. It is
# operationalised as this fixed preference order so the selection is deterministic and
# reproducible rather than resolved by whichever row pandas happened to sort first.
#
# The order is argued, not arbitrary: a K-Medoids cluster comes with a medoid that is an
# actual descriptor, so the factor can be named after something real. A hierarchical cut
# comes with a dendrogram and a threshold that reads directly as a correlation. HDBSCAN
# gives density structure but no representative and no threshold. KMeans and GMM give
# centroids in a rotated profile space that corresponds to nothing a chemist can point
# at, and GMM is additionally degenerate at this data shape -- see GMM_CONFIG.
METHOD_INTERPRETABILITY: tuple[str, ...] = (
    'kmedoids', 'hierarchical', 'hdbscan', 'kmeans', 'gmm',
)

# How many random candidates to draw for the "did selecting help?" baseline. The
# selected candidate's error is a minimum over 430 noisy estimates, so it flatters
# itself; the honest comparison is against what an arbitrary candidate would have
# scored. Cheap here because every candidate is scored anyway.
N_RANDOM_CANDIDATE_DRAWS: int = 200


# ---------------------------------------------------------------------------
# 6. Elastic Net  (Phases 10-12)
# ---------------------------------------------------------------------------

ELASTIC_NET_L1_RATIOS: list[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00]

# Logarithmic in alpha: the interesting behaviour is spread over orders of magnitude,
# and a linear grid would spend every point in the region where nothing is penalised.
ELASTIC_NET_ALPHA_MIN: float = 1e-5
ELASTIC_NET_ALPHA_MAX: float = 1e2
ELASTIC_NET_N_ALPHAS: int = 60

ELASTIC_NET_STABILITY_THRESHOLDS: list[float] = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# The plan's Step 10.3 order, among candidates within one SE of the best RMSE.
#
# 'alpha' sorts DESCENDING, which is the opposite direction from every other numeric
# tie-break in this project and is deliberate: the plan's last preference is "stronger
# regularization when otherwise equivalent", so among genuinely equivalent candidates
# the more penalised one wins.
#
# Selection stability and sign consistency are Phase 11 columns, so they are only
# available once stability has been computed and joined. elastic_net.select records
# which columns were actually applied rather than silently degrading.
ELASTIC_NET_TIE_BREAK: tuple[tuple[str, str], ...] = (
    ('n_nonzero_mean', 'lower'),                # fewest non-zero coefficients
    ('mean_selection_frequency', 'higher'),     # then selection stability
    ('mean_sign_consistency', 'higher'),        # then consistent signs
    ('alpha', 'higher'),                        # then stronger regularisation
)

OPTIMIZE_ELASTIC_NET_STABILITY_THRESHOLD: bool = True
ELASTIC_NET_STABILITY_THRESHOLD: float = 0.60


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """A configuration value cannot produce a valid feature set."""


def validate(known: tuple[str, ...] = cols.NUM_COLUMNS) -> list[str]:
    """Check every configured feature name against the descriptor vocabulary.

    Returns the list of non-fatal warnings (currently only duplicate cluster
    membership, which is legal but almost never intended). Raises ``ConfigError`` on
    anything that would produce a wrong or empty feature set.

    ``known`` is a parameter rather than a hardcoded reference so a test can validate
    a synthetic vocabulary without monkeypatching the module.
    """
    vocabulary = set(known)
    warnings: list[str] = []

    if not MANUAL_FEATURES:
        raise ConfigError('MANUAL_FEATURES is empty; nothing to select')

    missing = [f for f in MANUAL_FEATURES if f not in vocabulary]
    if missing:
        raise ConfigError(f'MANUAL_FEATURES names unknown descriptors: {missing}')

    duplicated = sorted({f for f in MANUAL_FEATURES if MANUAL_FEATURES.count(f) > 1})
    if duplicated:
        raise ConfigError(f'MANUAL_FEATURES repeats: {duplicated}')

    if cols.TARGET in MANUAL_FEATURES:
        raise ConfigError(f'MANUAL_FEATURES contains the target {cols.TARGET!r}')

    if not MANUAL_CLUSTERS:
        raise ConfigError('MANUAL_CLUSTERS is empty; no manual clusters to build')

    seen: dict[str, str] = {}
    for name, members in MANUAL_CLUSTERS.items():
        if not members:
            raise ConfigError(f'manual cluster {name!r} is empty')

        repeats = sorted({f for f in members if members.count(f) > 1})
        if repeats:
            raise ConfigError(f'manual cluster {name!r} repeats {repeats}')

        unknown = [f for f in members if f not in vocabulary]
        if unknown:
            raise ConfigError(f'manual cluster {name!r} names unknown descriptors: {unknown}')

        if cols.TARGET in members:
            raise ConfigError(f'manual cluster {name!r} contains the target {cols.TARGET!r}')

        for feature in members:
            if feature in seen:
                # Legal -- a feature can inform two factors -- but it makes the
                # factors correlated by construction, which is rarely what a hand
                # grouping means to say. Surfaced rather than raised.
                warnings.append(
                    f'{feature!r} is in both {seen[feature]!r} and {name!r}; '
                    'the two factors will be correlated by construction')
            else:
                seen[feature] = name

    if CLUSTER_FIT_SUBSET not in ('A', 'B'):
        raise ConfigError(
            f'CLUSTER_FIT_SUBSET is {CLUSTER_FIT_SUBSET!r}; expected "A" or "B". '
            'Subset C carries the target and must never fit an unsupervised transform.')

    if MRMR_N_FEATURES < 1:
        raise ConfigError(f'MRMR_N_FEATURES must be >= 1, got {MRMR_N_FEATURES}')

    unknown_methods = [m for m in CLUSTERING_METHODS if m not in REPRESENTATION_BY_METHOD]
    if unknown_methods:
        raise ConfigError(
            f'CLUSTERING_METHODS names algorithms with no declared representation: '
            f'{unknown_methods}')

    return warnings

