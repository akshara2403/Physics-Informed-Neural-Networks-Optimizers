**Empirical Solution-Gap and Stability Analysis of Matrix-Preconditioned
Optimizers for Physics-Informed Neural Networks**

*Literature Survey and Draft Introduction*

Prepared for Akshara -- draft for review and integration into the
manuscript. See `../RESULTS.md` for the full-scale (4 problem, 6
optimizer, 5 seed) results that supersede the "preliminary reduced-scale
trials" paragraph below.

**1. Literature Survey**

The 19 works below are organized into six clusters that map directly
onto this study's research question: the PINN framework itself, the
classical optimizers used as baselines, the modern matrix-preconditioned
optimizers under test, the body of work explaining why PINN loss
landscapes are hard to optimize, the benchmark-specific reference for
the Allen-Cahn equation, and the small number of very recent papers
that most directly overlap with this project's own angle.

| No. | Reference | Contribution | Relevance / Gap Addressed by This Study |
|---|---|---|---|
| **A. Foundational PINN Framework and Benchmarking Tools** | | | |
| 1 | [1] Raissi, Perdikaris & Karniadakis (2019) | Introduces the PINN framework: trains a neural network to satisfy a PDE residual, initial, and boundary conditions via automatic differentiation, without labeled solution data. | Defines the exact loss decomposition (L = L_PDE + L_BC + L_IC) and the heat/Burgers-style benchmark convention this study adopts as its evaluation setting. |
| 2 | [2] Lu, Meng, Mao & Karniadakis (2021) | Presents DeepXDE, a standardized library and benchmark suite for PINNs, including reference implementations for canonical PDEs. | Supplies the benchmarking conventions (domain sampling, reference-solution comparison) this study follows for fair, reproducible cross-optimizer comparison. |
| **B. Classical and Adaptive Optimizers Used as Baselines** | | | |
| 3 | [3] Kingma & Ba (2014) | Proposes Adam, a first-order optimizer using bias-corrected running averages of the gradient and squared gradient for adaptive per-parameter learning rates. | Adam is the de facto default optimizer in PINN training and the primary first-order baseline against which matrix-preconditioned methods are compared here. |
| 4 | [4] Liu & Nocedal (1989) | Develops L-BFGS, a limited-memory quasi-Newton method that approximates curvature from recent gradient history instead of forming the full Hessian. | L-BFGS and the Adam->L-BFGS hybrid are mandatory baselines in this study; this is the original method underlying PyTorch's LBFGS optimizer used in the experiments. |
| 5 | [5] Duchi, Hazan & Singer (2011) | Introduces AdaGrad, the first widely used per-coordinate adaptive learning-rate method, establishing the diagonal-preconditioning idea Adam later builds on. | Provides the historical/theoretical basis for why coordinate-wise adaptivity helps ill-conditioned objectives, motivating the extension to full-matrix preconditioning studied here. |
| **C. Matrix-Preconditioned and Structured Second-Order Optimizers** | | | |
| 6 | [6] Martens & Grosse (2015) | Proposes K-FAC, which approximates the Fisher/curvature matrix with a Kronecker-factored structure per layer to enable tractable second-order updates. | Establishes the structured (non-diagonal) preconditioning paradigm that Shampoo generalizes; motivates why curvature-aware updates could help PINNs' conflicting loss terms. |
| 7 | [7] Gupta, Koren & Singer (2018) | Introduces Shampoo, a matrix-preconditioned optimizer that maintains per-dimension preconditioners for tensor-shaped parameters via matrix roots. | Shampoo is the primary matrix-preconditioned optimizer evaluated in this study; its stability and cost on PINN loss landscapes is the central open question. |
| 8 | [8] Anil, Gupta, Koren, Regan & Singer (2020) | Scales Shampoo to large production models, addressing the computational cost of matrix-root preconditioning at scale. | Documents known numerical/engineering challenges (matrix inversion, conditioning) in Shampoo-style optimizers, directly relevant to the SVD-convergence failures observed in this study's pilot runs. |
| 9 | [9] Vyas et al. (2024) | Proposes SOAP, which runs Adam in Shampoo's preconditioner eigenbasis to improve stability and reduce Shampoo's overhead. | Positioned as an optional extension optimizer; shows preconditioning research is active and evolving, so this study frames its contribution as transferability testing rather than novelty of the optimizer itself. |
| 10 | [10] Liu et al. (2025) | Introduces Muon, a matrix-orthogonalization-based optimizer shown to be effective and scalable for large language model training. | Provides a second, architecturally distinct modern optimizer family (orthogonalized updates rather than matrix-root preconditioning) for comparison against Shampoo/SOAP. |
| **D. Why PINN Training Is Hard: Loss-Landscape and Failure-Mode Studies** | | | |
| 11 | [11] Krishnapriyan, Gholami, Zhe, Kirby & Mahoney (2021) | Shows PINNs can fail completely on simple-looking PDEs (e.g., convection-dominated regimes) even when the network has sufficient capacity, tracing failure to the optimization landscape rather than expressivity. | Directly motivates this study's hypothesis that optimizer choice, not just architecture, governs whether a PINN converges to an accurate solution. |
| 12 | [12] Wang, Teng & Perdikaris (2021) | Analyzes gradient statistics during PINN training and shows the PDE, boundary, and initial-condition loss terms produce imbalanced, conflicting gradients that stall standard optimizers. | Provides the mechanistic explanation for why a matrix-preconditioned optimizer that reshapes update directions across loss terms could plausibly help -- the core hypothesis tested here, and the direct motivation for this repository's new gradient-conflict-cosine diagnostic. |
| 13 | [13] Wang, Yu & Perdikaris (2022) | Uses a Neural Tangent Kernel analysis to explain why PINNs can fail to train, identifying spectral bias and unbalanced convergence rates across loss components. | Offers a theoretical account of PINN ill-conditioning that this study treats as background motivation, without claiming its empirical results prove or disprove any NTK-based theorem. |
| 14 | [14] McClenny & Braga-Neto (2022/2023) | Proposes self-adaptive PINNs that learn per-point loss weights via a soft-attention mechanism to counter gradient imbalance. | Represents an alternative (loss-reweighting) mitigation strategy to the optimizer-level approach studied here; clarifies that this paper isolates the optimizer as the sole intervention, holding the loss formulation fixed. |
| 15 | [15] Wang, Sankaran & Perdikaris (2022) | Shows that causally respecting the temporal ordering of PDE residuals during training substantially improves PINN convergence on time-dependent problems. | Identifies training-schedule design as another confound; this study controls for it by using identical collocation/seed protocols across all optimizers so that only the optimizer varies. |
| **E. Benchmark-Specific Reference Work** | | | |
| 16 | [16] Wight & Zhao (2020) | Solves the Allen-Cahn and Cahn-Hilliard equations with adaptive PINNs, documenting the stiffness and sharp-interface difficulty of Allen-Cahn as a PINN benchmark. | Justifies this study's choice of Allen-Cahn as one of the hardest benchmarks and confirms that a stiff finite-difference reference solution (as used here) is an accepted ground truth; the difficulty this paper documents is directly reproduced in this repository's own results (see RESULTS.md). |
| **F. Optimizer-for-PINN Studies Most Directly Related to This Work** | | | |
| 17 | [17] Kiyani, Shukla, Urban, Darbon & Karniadakis (2025) | Systematically compares first-order, quasi-Newton, and self-scaled quasi-Newton optimizers on challenging PDE benchmarks (Burgers, Allen-Cahn, Kuramoto-Sivashinsky, Ginzburg-Landau). | The closest prior comparative study; this paper extends its scope specifically to matrix-preconditioned optimizers (Shampoo/SOAP/Muon), which are not covered there. |
| 18 | [18] Wang, Bhartari, Li & Perdikaris (2025) | Analyzes PINN optimization from a gradient-alignment, second-order perspective, connecting curvature properties of the PINN loss to convergence behavior. | Supplies theoretical grounding for why second-order/preconditioned methods may interact differently with PINN loss geometry than with standard deep-learning losses; directly parallels this repository's Hutchinson-trace curvature diagnostic. |
| 19 | [19] Churiwala, Shukla & Khullar (2025) | Proposes a PDE-aware optimizer that adapts its update rule to PDE-residual structure, an explicit example of preconditioning research already active in the PINN literature. | Confirms that PDE-aware/preconditioning ideas are already emerging in PINN optimization, so this study is framed as an empirical transferability and stability analysis rather than a first-of-its-kind claim. |

**2. Introduction (Draft)**

Physics-Informed Neural Networks (PINNs) have emerged as a widely used
framework for approximating solutions of partial differential equations
(PDEs) by embedding the governing physics directly into the training
loss of a neural network [1]. Rather than learning from a large
labeled dataset, a PINN is trained so that its output u_theta(x, t) satisfies
the PDE residual, together with the initial and boundary conditions, at
a set of collocation points sampled from the problem domain. This
formulation has made PINNs attractive across computational physics and
engineering, and software libraries such as DeepXDE have further
standardized their implementation and benchmarking [2]. However, the
practical reliability of PINNs continues to depend heavily on a factor
that is often treated as an implementation detail rather than a research
question in its own right: the choice of optimizer used to train the
network.

Unlike conventional supervised learning, the PINN loss is a weighted sum
of physics, boundary, and initial-condition terms, L = L_PDE + L_BC +
L_IC, whose components can differ by several orders of magnitude and
pull the network parameters in conflicting directions. A growing body of
work has shown that this structure produces loss landscapes that are
ill-conditioned and exhibit unbalanced gradients across loss terms,
which can trap standard optimizers in states that satisfy the PDE
residual locally while remaining far from the true solution [11],
[12], [13]. Related studies have shown that PINNs can fail to train
altogether on seemingly simple problems once the physical regime becomes
sufficiently stiff or convection-dominated [11], and have proposed
loss-balancing, causality-respecting, and self-adaptive weighting
schemes to mitigate these pathologies [14], [15]. These findings
motivate treating optimization, not just network architecture or loss
design, as a first-class variable in PINN research.

In practice, PINN training almost universally relies on Adam [3], a
first-order adaptive-moment optimizer, or L-BFGS [4], a limited-memory
quasi-Newton method that approximates curvature information without
forming the full Hessian, or a two-phase hybrid in which Adam performs
rapid early-stage descent before L-BFGS refines the solution near
convergence. This Adam-to-L-BFGS pipeline has become close to a de facto
baseline in the PINN literature, yet Adam and L-BFGS were designed for
generic non-convex and offline unconstrained optimization respectively,
not for the specific multi-term, physics-constrained loss geometry that
PINNs induce.

Meanwhile, large-scale deep learning has increasingly moved toward
matrix-preconditioned and structured second-order optimizers that
reshape parameter updates using curvature or covariance information
along tensor dimensions, rather than a single scalar or per-coordinate
learning rate. Kronecker-factored approximate curvature (K-FAC) [6]
and Shampoo [7], together with its scaled implementation for large
models [8], estimate block- or Kronecker-structured preconditioners
that can straighten narrow, ill-conditioned optimization valleys. More
recently, SOAP [9] has been proposed to stabilize Shampoo by combining
it with Adam-style adaptivity in the preconditioner eigenbasis, and Muon
[10], based on matrix-orthogonalized updates, has shown promising
results in large language model training. These methods were developed
primarily to accelerate training of large-scale vision and language
models; whether their curvature-aware update rules transfer to the very
different, PDE-residual-driven loss geometry of PINNs is comparatively
unexplored, though a small number of very recent studies have begun to
probe this question directly [17], [18], [19].

Existing PINN optimizer comparisons largely stop at reporting which
method reaches the lowest training loss for a single benchmark and a
single random seed, which conflates optimization speed with solution
accuracy and can overstate the reliability of a given optimizer. This
paper instead asks a more specific question: do modern
matrix-preconditioned optimizers from large-scale deep learning reduce
empirical solution gap, convergence instability, runtime-normalized
error, and seed sensitivity in PINNs, relative to Adam, L-BFGS, and the
standard Adam+L-BFGS baseline? Answering this requires evaluating
optimizers not only against final training loss, but against the actual
PDE solution, and doing so across multiple random seeds so that
stability -- not just a single favorable run -- can be assessed.

Because the true global minimizer theta* of the PINN loss is not known in
practice, this paper avoids the stronger and unverifiable claim of
measuring an "optimality gap," L(theta) - L(theta*). Instead, we report the
empirical relative solution gap, ||u_true - u_pred||_2 / ||u_true||_2,
computed against exact or high-accuracy reference solutions, as an
empirical proxy for how closely a trained PINN approximates the true PDE
solution. Low PDE residual is reported alongside this metric but is not
treated as proof of solution correctness on its own.

To this end, an identical fully connected network architecture (2 ->
32 -> 32 -> 32 -> 1) is trained under matched collocation, initial-, and
boundary-point sampling across benchmarks of increasing difficulty: the
1D heat equation, the nonlinear viscous Burgers' equation [1], the
stiff, nonlinear Allen-Cahn equation [16], and (new in this study,
beyond the original roadmap) the Korteweg-de Vries equation, whose
third-order-derivative residual specifically stresses the
higher-order-curvature sensitivity that motivates matrix preconditioning
in the first place. Each benchmark is evaluated against exact or
numerically generated finite-difference/analytic reference solutions.
Adam, L-BFGS, the Adam+L-BFGS hybrid, and three from-scratch
matrix-preconditioned/orthogonalized optimizers -- Shampoo, SOAP, and
Muon -- are each trained under multiple random seeds and evaluated on
relative solution gap, PDE residual, boundary/initial-condition error,
wall-clock runtime, runtime-normalized error, loss oscillation,
gradient-conflict cosine, Hutchinson curvature trace, and explicit
failure flags for divergence or non-finite loss.

Full-scale results (4 benchmark problems, 6 optimizers, 5 random seeds
each) are reported and analyzed in `RESULTS.md`, alongside every
methodological caveat -- CPU-only compute budget, per-problem step
counts, and low-statistical-power warnings on any test computed from
fewer than 8 seeds -- needed to read those numbers responsibly. An
earlier, much smaller pilot (CPU, two seeds, 1,500 collocation points,
two benchmarks, and a third-party Shampoo implementation that failed
outright via SVD non-convergence) motivated this study's from-scratch,
numerically stabilized reimplementation of Shampoo and is documented for
provenance in the repository's README rather than reported as a result
in its own right.

The contributions of this study are therefore: (1) an evaluation of
optimizer transferability from large-scale deep learning to PINNs that
uses solution accuracy, not training loss, as the primary criterion; (2)
explicit measurement of seed sensitivity and failure behavior alongside
accuracy, rather than a single best-seed result; (3) inclusion of
Adam+L-BFGS as a mandatory, literature-standard baseline so no
comparison looks artificially favorable; (4) two new diagnostics
(gradient-conflict cosine, Hutchinson curvature trace) that connect
observed optimizer behavior back to the specific loss-landscape
pathologies the literature has identified as the reason PINN training is
hard; and (5) a reproducible, tested benchmark suite, training protocol,
and results archive released alongside this paper. The remainder of
this paper is organized as follows: Section II reviews related work on
PINN training difficulty and optimizer design in greater detail; Section
III describes the network architecture, benchmark PDEs, and training
protocol; Section IV presents the empirical results; and Section V
discusses their implications, limitations, and directions for future
work.

**References**

[1] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," Journal of Computational Physics, vol. 378, pp. 686-707, 2019.

[2] L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis, "DeepXDE: A deep learning library for solving differential equations," SIAM Review, vol. 63, no. 1, pp. 208-228, 2021.

[3] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv:1412.6980, 2014.

[4] D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization," Mathematical Programming, vol. 45, no. 1, pp. 503-528, 1989.

[5] J. Duchi, E. Hazan, and Y. Singer, "Adaptive subgradient methods for online learning and stochastic optimization," Journal of Machine Learning Research, vol. 12, pp. 2121-2159, 2011.

[6] J. Martens and R. Grosse, "Optimizing neural networks with Kronecker-factored approximate curvature," in Proc. 32nd Int. Conf. on Machine Learning (ICML), PMLR vol. 37, pp. 2408-2417, 2015.

[7] V. Gupta, T. Koren, and Y. Singer, "Shampoo: Preconditioned stochastic tensor optimization," arXiv:1802.09568, 2018.

[8] R. Anil, V. Gupta, T. Koren, K. Regan, and Y. Singer, "Scalable second order optimization for deep learning," arXiv:2002.09018, 2020.

[9] N. Vyas et al., "SOAP: Improving and stabilizing Shampoo using Adam," arXiv:2409.11321, 2024.

[10] J. Liu et al., "Muon is scalable for LLM training," arXiv:2502.16982, 2025.

[11] A. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, and M. W. Mahoney, "Characterizing possible failure modes in physics-informed neural networks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021; arXiv:2109.01050.

[12] S. Wang, Y. Teng, and P. Perdikaris, "Understanding and mitigating gradient flow pathologies in physics-informed neural networks," SIAM Journal on Scientific Computing, vol. 43, no. 5, pp. A3055-A3081, 2021.

[13] S. Wang, X. Yu, and P. Perdikaris, "When and why PINNs fail to train: A neural tangent kernel perspective," Journal of Computational Physics, vol. 449, p. 110768, 2022; arXiv:2007.14527.

[14] L. McClenny and U. Braga-Neto, "Self-adaptive physics-informed neural networks using a soft attention mechanism," Journal of Computational Physics, vol. 474, p. 111722, 2023; arXiv:2009.04544.

[15] S. Wang, S. Sankaran, and P. Perdikaris, "Respecting causality is all you need for training physics-informed neural networks," arXiv:2203.07404, 2022.

[16] C. L. Wight and J. Zhao, "Solving Allen-Cahn and Cahn-Hilliard equations using the adaptive physics informed neural networks," arXiv:2007.04542, 2020.

[17] E. Kiyani, K. Shukla, J. F. Urban, J. Darbon, and G. E. Karniadakis, "Which optimizer works best for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks?" arXiv:2501.16371, 2025.

[18] S. Wang, A. K. Bhartari, B. Li, and P. Perdikaris, "Gradient alignment in Physics-informed Neural Networks: A second-order optimization perspective," arXiv:2502.00604, 2025.

[19] V. Churiwala, H. Shukla, and M. Khullar, "PDE-aware optimizer for Physics-informed Neural Networks," arXiv:2507.08118, 2025.
