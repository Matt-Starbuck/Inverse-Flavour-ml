# Machine Learning for the Flavour Puzzle
PyTorch framework that learns Beyond the Standard Model theories that can compress quark yukawa singular values by the Planck scale, while remaining compatible with low energy constraints. 

## Key Results
Given a new physics scale of $\Lambda \in \left[3\text{TeV}, 100\text{TeV}\right]$, the trained neural network outputs complex BSM couplings $\left(\lambda_{q}, \lambda_{u}, \lambda_{d}\right)$ that modify the quark Yukawa matrices up to the Planck scale ($M_{\text{Pl}} = 10^{19} \text{GeV}$)

### Evolution of Yukawa Couplings from $\Lambda =  3 \text{TeV}$
Starting from the 5 order of magnitude Standard Model hierarchy at $3\text{TeV}$, the learned couplings drive the singular values to $\mathcal{O}(1)$ of each other by $M_{\text{Pl}}$:

![RG Evolution](src/training/plots/rg_evolution.png)

### Hierarchy Compression
Across the continuous range of BSM scales $\Lambda \in \left[3\text{TeV}, 100\text{TeV}\right]$, the network compresses adjacent generation ratios from $\mathcal{O}(10^{2}-10^{3})$ down to $\mathcal{O}(1)$ by the Planck scale.

![Hierarchy Loss](src/training/plots/hierarchy_loss_vs_lambda.png)

### Experimental Bounds

## Background for non-physicists:
Protons are made of two up quarks and one down quark. We know, however, of six different kinds of quarks in nature: up, down, charm, strange, top and bottom. The masses of the quarks are extremely hierarchical in nature, which the top quark being around 5 order of magnitude heavier than the up quark. A huge question in theoretical physics today is why this hierarhcy exists. 

## Assumptions of this work
A few assumptions have been made about the BSM theories considered
* The BSM couplings do not run themselves. This is to make the numerics tractable.
* The 
