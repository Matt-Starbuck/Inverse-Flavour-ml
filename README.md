# Machine Learning for the Flavour Puzzle
PyTorch framework that learns Beyond the Standard Model theories that can compress quark yukawa singular values by the Planck scale, while remaining compatible with low energy constraints. 

## Key Results
Given a new physics scale of $\Lambda \in \left[3\text{TeV}, 100\text{TeV}\right]$, the trained neural network outputs complex BSM couplings $\left(\lambda_{q}, \lambda_{u}, \lambda_{d}\right)$ that modify the quark Yukawa matrices up to the Planck scale ($M_{\text{Pl}} = 10^{19} \text{ GeV}$).

### Evolution of Yukawa Couplings from $\Lambda =  3 \text{TeV}$
Starting from the 5 order of magnitude Standard Model hierarchy at $3\text{ TeV}$, the learned couplings drive the singular values to $\mathcal{O}(1)$ of each other by $M_{\text{Pl}}$:

![RG Evolution](src/training/plots/rg_evolution.png)

### Hierarchy Compression
Across the continuous range of BSM scales $\Lambda \in \left[3\text{ TeV}, 100\text{ TeV}\right]$, the network compresses adjacent generation ratios from $\mathcal{O}(10^{2}-10^{3})$ down to $\mathcal{O}(1)$ by the Planck scale.

![Hierarchy Loss](src/training/plots/hierarchy_loss_vs_lambda.png) 

### Experimental Bounds
Conservatively assuming that the $\left(\lambda_{q}, \lambda_{u}, \lambda_{d}\right)$ couplings can mediate four-quark operators at tree-level, EFT coefficients are computed, and constrained by $95\%$ confidence level bounds taken from (https://arxiv.org/pdf/2402.09503). In all cases the model predicts parameters that are compatible with these bounds.

![Pheno Loss](src/training/plots/pheno_loss_vs_lambda.png)

**Physical Insights (so far):**
* Strong couplings within the up sector $\text{max}\left(\lambda_{u}\right) \sim 3$ preferred.
* Contact operator bounds appear to have a greater intrinsic tension with generating the hierarhcy than FCNC bounds do. 

## Background for non-physicists:

The Standard Model of particle physics (SM) describes the interactions between all of the elementary particles that make up the universe. In the SM, the proton and neutron are composed of quarks, of which there are six, three 'up types' (up, charm, top), and three 'down types' (down, strange, bottom). One of the deepest open questions within the SM, known as the **flavour puzzle**, is why their masses span five orders of magnitude, from the up quark ($\sim 1\text{ MeV}$) to the top quark ($\sim 100\text{ GeV}$). 

The quark masses are governed by the singular values of two 3x3 complex 'Yukawa' matrices ($Y_{u}, Y_{d}$). These matrices are not constant, but depend on the energy scale at which the experiment that measures them is done, $\mu$, and therefore the quark masses are predicted to change with energy scale. The differential equations which describe how the yukawa matrices change with energy scale are called the renormalisation group equations (RGEs), and have the form

$\frac{dY_{u}}{d\ln\mu} = \beta_{u}\left(\mu\right)$ 

$\frac{dY_{d}}{d\ln\mu} = \beta_{d}\left(\mu\right)$.

One possible solution to the flavour problem is that at very high energy scales, the singular values of the Yukawa matrices evolve to be comparable to each other. In the SM, this is known (and can be tested in principle with this code) not to happen. However, it is possible that in a more fundamental theory of particle physics than the SM, the $\beta_{u}\left(\mu\right)$ and $\beta_{d}\left(\mu\right)$ functions could be modified so that this does occurr. In the literature, much investigation has gone into starting with a model where the singular values are comparable, and predicting the observed Yukawa matrices at low energies. Whilst many models have been found where this works, they often lead to severe tensions with other experimental constraints. 

**Goal of this project: The inverse problem.** Starting from the known Yukawa matrices at low energy scales, can a neural network find the $\beta_{u}\left(\mu\right)$ and $\beta_{d}\left(\mu\right)$ functions that compress the singular values at high energies whilst remaining compatible with these low energy constraints? 

The neural network takes in a scale of new physics $\Lambda$ as input, and outputs three 3x3 complex matrices $\left(\lambda_{q}, \lambda_{u}, \lambda_{d}\right)$ which define (by assumption, see below) the RGEs above the scale $\Lambda$ as

$\beta_{u} = \beta_{u}^{\text{SM}} + \frac{1}{16 \pi^{2}}\lambda_{q} \lambda_{u}^{\dagger}$

$\beta_{d} = \beta_{d}^{\text{SM}} + \frac{1}{16 \pi^{2}}\lambda_{q} \lambda_{d}^{\dagger}$

The $\left(\lambda_{q}, \lambda_{u}, \lambda_{d}\right)$ matrices then enter a multi-objective loss function to be described below.

## Differentiable Pipeline and Multi-Objective Loss

Once the $\left(\lambda_{q}, \lambda_{u}, \lambda_{d}\right)$ matrices have been predicted by the neural network, they are fed into a three component loss function.

1. **Phenomenological Constraints:** The matrices are used to make predictions for EFT Wilson coefficients ($C \sim \lambda^{4}/\Lambda^{2}$), and the ratio of the prediction to the maximum allowed value is recorded in a dictionary (src/loss_functions/eft_bounds.py).

2. **Hierarchy Compression by $M_{\text{Pl}}$:** The Yukawa couplings are batch evolved to the scales $\Lambda$ (src/inverse_flavour/get_initial_bsm.py). These Yukawa couplings are batch evolved from the $\Lambda$ scales up to $M_{\text{Pl}}$ (src/physics/bsm_rge.py), and the ratios of singular values are computed and recorded in a dictionary.

3. **Perturbativity Bounds:** The $\left(\lambda_{q}, \lambda_{u}, \lambda_{d}\right)$
are exponentially punished for having couplings greater than $2\pi$.

## Repository Structure

├── src/
│   ├── inverse_flavour/   # Initial Parameter values at $\mu=3\text{ TeV}$.
│   ├── physics/           # Vectorized SM & BSM 1-loop RGE ODE modules
│   ├── loss_functions/    # Planck SVD hierarchy loss and EFT meson/dijet bounds
│   ├── models/            # Deterministic MLP architecture & complex tensor reshaping
│   └── training/          # Scale sampling, training loop, and diagnostic plotting
└── tests/                 # Test of end-to-end forward/backward workflow.


## Physics Simplifications and Assumptions (v1.0)
In this first version, a few assumptions have been made about the new physics that modifies the RGEs of Yukawa couplings
1. **The BSM couplings $\left(\lambda_{q}, \lambda_{u}, \lambda_{d}\right)$ do not run themselves**. In reality, unless the new physics is very special, these couplings will vary with scale. This choice was made to keep the dimension of the space of new physics theories tractable, and will be investigated in v2.0

2. **The BSM physics gives corrections to the vertex**. This assumption is made so that the BSM contribution to $\beta_{u}$ and $\beta_{d}$ does not involve $Y_{u}$ and $Y_{d}$ respectively. If this assumption is not made, the singular value ratio evolution does not differ significantly from that of the SM.

3. **The right handed up and down sector corrections are independent**. This way $\lambda_{u}$ does not enter $\beta_{d}$ and $\lambda_{d}$ does not enter $\beta_{u}$.

## Quick Start
### Run the physics & autograd verification suite
python -m tests.test_non_ML_workflow

### Train the deterministic model
python -m src.training.train

### Generate Planck hierarchy, EFT bound, and RG trajectory plots
python -m src.training.predictions