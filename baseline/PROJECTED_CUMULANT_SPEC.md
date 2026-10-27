# Двовимірна кумулянтна апроксимація ReLU-ядра

2026-10-24. Протокол зафіксовано до реалізації та результатів.
Статус: окремий feasibility-пілот; попередні frozen evidence не змінюються.

## Питання і метод

K(x,y)=E[(W·x)_+(W·y)_+], iid standardized Gamma(1) або Laplace.
Для одиничних x,y та rho=x·y, |rho|<1, a=x,
b=(y-rho*x)/sqrt(1-rho²) маємо Gaussian reference у незалежних координатах.
D3=sum_{p+q=3} κ3(W)sum_j a_j^p b_j^q/(p!q!) ∂1^p ∂2^q;
D4 аналогічно. Кандидати: G; G+D3; G+D3+D4;
G+D3+D4+D3²/2. Останній включає Hermite order6 від squared skewness,
але не κ6: це розклад другого порядку для делокалізованих проєкцій.
Інтеграли E[ReLU(Z1)ReLU(rho Z1+s Z2) Hp(Z1)Hq(Z2)]
обчислюються аналітично через truncated Gaussian moments.
На діагоналі використовується одновимірна границя; антипод має K=0.
Жодного clipping/PSD repair у primary. Окремо фіксуються від'ємні entries/eigenvalues.
Це апроксимація, не нова доведена uniform Edgeworth-теорема для ReLU.

## Дизайн

Dimensions 4,16,32,64; 8 inputs/cell; laws Gamma(1), Laplace.
Три геометрії, deterministic RNG seed2026102401+d:
- dense: normalized Gaussian directions;
- sparse: normalized directions with only first2 nonzero coordinates;
- near: dense directions with near-parallel and near-antiparallel pairs,
  angular perturbation .001; includes exact duplicate and antipode.
Звичайний Gaussian kernel — обов'язковий baseline.
Sobol N=256,1024,4096,16384,65536, 8 independent scrambles.
Conditional Sobol N=256,1024,4096: точне інтегрування останньої координати;
для sparse — координати1, для інших — останньої. Zero coefficients
обробляються явно, без division by zero.

## Reference і чесність бюджету

Independent scrambled direct Sobol reference: 8×2^15 coarse та8×2^17 fine.
Якщо 3 Frobenius SE > .0003 або coarse/fine difference >3 combined SE,
одне наперед задане підвищення до8×2^19. Seed families reference separate
from method scrambles. Записати endpoint guards. Reference uncertainty
емпірична, без deterministic guarantee. Targets .01,.003,.001;
reference-adequate лише якщо3SE <=target/10 і convergence check PASS.
Складні випадки не вилучаються. Sparse reference додатково перевіряється
незалежним 1D conditional adaptive quadrature, коли практично можливо.

Час: single BLAS thread; 1 warmup, median5 evaluations; node generation
в Sobol/conditional включена. Cumulant includes geometry and cumulants;
жодного hidden precomputation. Record matrices, all timing repeats,
reference replicates, versions and hashes. Немає підбору напряму whitening
або truncation order по test: усі варіанти показати, primary=G+D3+D4+D3²/2.

## Gates

Technical: exact Gaussian case; independent adaptive-integral check of
Hermite contractions; x/y swap; input permutations; scaling; duplicate,
antipode; direct algebra of cumulant contractions; eigenvalues.
Scientific: primary reduces error vs Gaussian by>=20% у обох laws
щонайменше на двох d>=16 dense cells per law; absolute accuracy separately.
Practical: at target .003, certified upper empirical error+reference
allowance meets target, PSD within1e-10 relative tolerance, median total
runtime <=.8 best noncalibrated comparator that meets same target;
at least2 distinct (law,d) dense cells. Near/sparse are explicit robustness
stress tests; broad claim requires all geometries. Strict comparisons
use worst observed Sobol error plus reference allowance; also save RMSE
for descriptive analysis. Timing claims are environment-specific.

Stop after this pilot with explicit result. No training benefit/novelty claim.
