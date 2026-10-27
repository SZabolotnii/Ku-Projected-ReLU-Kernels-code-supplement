# Post-primary уточнення еталона

2026-10-24. Записано після primary результатів, до додаткового прогону.
Primary reference на d>=16 переважно не розділяє target .003; це
reference-limited результат, а не встановлений провал практичної гіпотези.
Первинні arrays/rows/timings/manifest зберігаються незмінними.

Новий еталон: randomized QMC із точно центрованим quadratic control.
Для U=W·x,V=W·y, ||x||=||y||=1, rho=x·y:
C=a(U+V)+b(U²+V²−2)+c(UV−rho),
a=(1+rho)/(2sqrt(2pi)), b=sqrt(1−rho²)/(4pi),
c=1/4+arcsin(rho)/(2pi). EC=0 для будь-якого iid центрованого
unit-variance закону. Коефіцієнти фіксовані з Gaussian degree<=2
проєкції; вони не потребують pilot і не використовують κ3/κ4-кандидат.
Еталон = sample mean(g−C). Контроль обчислюється через sample means
та sample covariance проєкцій. Це не зміна оцінюваного kernel.

Для всіх попередніх cells:8×2^17 coarse і8×2^19 fine, seeds510M/610M.
За 3SE>.0001 або convergence fail одне підвищення8×2^21 seed710M.
Залишаються ті самі reference gates; unresolved cases зберігають REF-LIMIT.
Порівняти refined із первинним direct reference у3 combined SE.
Sparse cells перевірити незалежною 1D adaptive conditional quadrature.

Переоцінити лише errors/targets тих самих збережених kernels і timings;
метод, geometry, competitors, gates не змінюються. Primary звіт і
post-primary уточнення показувати окремо. Статистичні межі емпіричні.
