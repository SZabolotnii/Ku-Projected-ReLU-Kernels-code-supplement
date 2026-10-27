"""Generate completion audit and manifests for derived projected-kernel artifacts."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'evidence/projected_resolution'
    a=json.loads((folder/'analysis.json').read_text())
    v=json.loads((folder/'verification.json').read_text())
    prior=json.loads((ROOT/'evidence/projected_refined/verification.json').read_text())
    pre=json.loads((ROOT/'evidence/projected_cumulant/preflight.json').read_text())
    text=['# Завершення projected-cumulant пілота', '\n2026-10-24.',
          '\n## Результат',
          f"\nМеханістичний критерій: {'PASS' if a['scientific_gate'] else 'FAIL'}. Практичний критерій при target .003: {'PASS' if a['practical_gate'] else 'FAIL'}.",
          '\nПісля незалежного уточнення еталона та додавання quadratic-CV Sobol '
          'кваліфікуються dense cells: '+', '.join(f'{law}, d={d}' for law,d in a['practical_cells'])+'.',
          '\nЦе позитивний feasibility-сигнал для аналітичної кумулянтної '
          'апроксимації першошарового ReLU-ядра за відомого iid закону ваг. '
          'Він не змінює негативні висновки попереднього HL/calibration напряму.',
          '\n## Що саме виконано',
          '\n- Аналітичне інтегрування Hermite terms порядків3/4/6 у двох проєкціях; '
          'degree6 походить від κ3². Повний виклад: [PROJECTED_DERIVATION_UA.md](PROJECTED_DERIVATION_UA.md).',
          '\n- Primary protocol зафіксовано до коду/результатів; 1632 rows, '
          'd4/16/32/64, Gamma(1)/Laplace, три геометрії,8 inputs/panel.',
          '\n- Primary scientific PASS; practical gate спершу не пройдено через '
          'поєднання недостатньої точності та reference-limited cells. '
          'Два post-primary reference extensions мають окремі SPEC і evidence; '
          'початкові kernels/timings не змінені. Останній budget8×2^23 тільки '
          'для d32/d64; усі target .003 reference gates тепер проходять.',
          '\n- Додано288 rows quadratic-CV Sobol як сильнішого конкурента. '
          'Збережено всі seeds, матриці, часові повтори, PSD failures, reference '
          'реплікації та SHA256. Час еталона не є вартістю використання методу.',
          '\n## Незалежні перевірки',
          f"\n- Preflight {pre['passed']}/{pre['total']}; числове nested integration, "
          'симетрії, масштабування, Gaussian і singular limits.',
          f"\n- Primary/refined verification {prior['passed']}/{prior['total']}; "
          'усі stored metrics/costs, усі deterministic matrices, representative '
          'stochastic replay,8 coarse reference scrambles та sparse adaptive quadrature.',
          f"\n- Final reference/strong controls verification {v['passed']}/{v['total']}; "
          'усі288 comparator metrics, representative seed replay, reference means '
          'і diagnostic recomputation, independent sparse quadrature, old/direct parity.',
          '\n- Повний повтор усіх timed runs і final8×2^23 references не виконувався '
          'і не заявляється. Емпіричні3SE не є доведеною довірчою межею.',
          '\n## Межі, які не можна прибирати',
          '\n1. Один fixed panel із8 входів на geometry/d. QMC seeds не є '
          'незалежними datasets. Потрібна перевірка на нових input panels '
          'і більших kernel matrices перед узагальненням.',
          '\n2. Sparse cells не досягають .003; Gamma(1),d32,sparse має '
          'від’ємне власне число. PSD repair не використовувався.',
          '\n3. Для Laplace,d64,dense звичайний Gaussian уже досягає .003 '
          'і швидший за поправку. Немає універсальної переваги.',
          '\n4. Time-to-target — ретроспективний результат на дискретній сітці N, '
          'із strict worst observed error для8 scrambles. Сильніший comparator '
          'виміряно окремою сесією. Великі speed ratios не переносити на '
          'будь-який бюджет, реалізацію або обладнання.',
          '\n5. Gaussian–Hermite/Edgeworth механізм відомий; новизну конкретної '
          'реалізації/теорії ще не встановлено. Немає adaptive PATP benefit, '
          'теореми uniform remainder для ReLU, downstream prediction experiment '
          'або finite-width deep CNN результату.',
          '\n## Артефакти',
          '\n[Фінальний звіт](PROJECTED_RESOLVED_REPORT_UA.md), '
          '[первинний звіт](PROJECTED_CUMULANT_REPORT_UA.md), '
          '[перше уточнення](PROJECTED_REFINED_REPORT_UA.md).',
          '\n[Час–точність](evidence/projected_resolution/time_accuracy.png), '
          '[геометрія входів](evidence/projected_resolution/geometry_accuracy.png).']
    (ROOT/'PROJECTED_COMPLETION_AUDIT.md').write_text('\n'.join(text)+'\n')
    for name,report in (('projected_cumulant','PROJECTED_CUMULANT_REPORT_UA.md'),('projected_refined','PROJECTED_REFINED_REPORT_UA.md'),('projected_resolution','PROJECTED_RESOLVED_REPORT_UA.md')):
        out=ROOT/'evidence'/name
        files=[out/n for n in ('analysis.json','summary.csv','comparisons.csv')]
        if (out/'verification.json').exists():files.append(out/'verification.json')
        files+=[ROOT/report,ROOT/'scripts/projected_report.py']
        if name=='projected_refined':files.append(ROOT/'scripts/projected_verify.py')
        if name=='projected_resolution':
            files += [ROOT/'PROJECTED_COMPLETION_AUDIT.md',ROOT/'PROJECTED_DERIVATION_UA.md']
            files += [ROOT/'scripts'/s for s in ('projected_resolve_verify.py','projected_plot.py','projected_finalize.py')]
            files += [out/n for n in ('time_accuracy.png','time_accuracy.pdf','geometry_accuracy.png','geometry_accuracy.pdf')]
        manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Completion audit and three analysis manifests generated.')
if __name__=='__main__':main()
