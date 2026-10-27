# Gamma d64: post-primary уточнення незалежних панелей

2026-10-25. Зафіксовано до нового прогону. Попередній generalization результат і його NOT PASSED зберігаються.

Ті самі три панелі 32×64, та сама second_order формула. Нові незалежні quadratic-CV Sobol еталони: 8 scrambles ×2^23; якщо 3 Frobenius SE>.0003 або різниця з попередніми 8×2^22 перевищує 3 combined SE, виконати 8×2^24. Після цього жорстка зупинка REFERENCE_LIMITED. Seeds=50000000+panel*100+level*10+rep (level=0,1). Новий рівень порівнюється з безпосередньо попереднім. Старі еталони не домішуються до нового середнього.

Усі candidate та competitor матриці й timings беруться незмінними з projected_generalization. Перераховуються лише похибки та рішення. Критерії ті самі: target .003, allowance=(maxerror+u)/(1-u), PSD>=-1e-10 у нормованих одиницях; практична перевага при time ratio<=.8. Якщо жоден конкурент не досягає цілі — COMPETITOR_LIMITED, без твердження про числовий speedup. Нових competitor budgets не додаємо.

Підсумок по трьох Gamma-панелях та окремо шести панелях Gamma64+Laplace16, останні успадковані з попереднього аналізу. Це post-primary закриття reference uncertainty, не новий незалежний confirmatory дизайн і не перевірка новизни. Known laws, unshifted ReLU, перший шар, dense входи; попередні негативні результати незмінні.

Зберегти всі репліки corrected/direct, inputs, seeds, guards, hashes. Верифікація hash усіх залежностей, перерахунок reference diagnostics, всіх 198 похибок і PSD, deterministic replay; парне corrected/direct порівняння діагностичне. Еталон повністю вдруге не повторюється. Час еталона не входить у час методу.
