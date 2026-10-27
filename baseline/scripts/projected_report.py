"""Derive all scientific/practical summaries from immutable primary evidence."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'evidence/projected_cumulant'

def analyze(refined=False,resolved=False):
    rows=list(csv.DictReader((OUT/'raw.csv').open()))
    reference_dir=ROOT/'evidence/projected_resolution' if resolved else ROOT/'evidence/projected_refined' if refined else OUT
    if resolved:
        rows+=list(csv.DictReader((reference_dir/'raw.csv').open()))
    diag=json.loads((reference_dir/'reference_diagnostics.json').read_text())
    if refined or resolved:
        kernels=dict(np.load(OUT/'kernels.npz')); references={}
        if resolved: kernels.update(dict(np.load(reference_dir/'kernels.npz')))
        for row in rows:
            cell=f"{row['law']}_d{row['d']}"
            if cell not in references:
                rp=reference_dir/f'reference_{cell}.npz'
                if not rp.exists(): rp=ROOT/'evidence/projected_refined'/rp.name
                references[cell]=np.load(rp)['mean']
            ref=references[cell][('dense','sparse','near').index(row['geometry'])]
            row['error']=float(np.linalg.norm(kernels[row['key']]-ref)/np.linalg.norm(ref))
            row['mineig']=float(np.linalg.eigvalsh(kernels[row['key']])[0]/np.linalg.norm(ref))
    groups={}
    for row in rows:
        for key in ('d','n','rep'): row[key]=int(row[key])
        for key in ('error','seconds','mineig','minentry'): row[key]=float(row[key])
        key=(row['law'],row['d'],row['geometry'],row['method'],row['n'])
        groups.setdefault(key,[]).append(row)
    summary=[]
    for (law,d,geo,method,n),rr in groups.items():
        ref=diag[f'{law}_d{d}'][('dense','sparse','near').index(geo)]; u=ref['three_se']
        summary.append(dict(law=law,d=d,geometry=geo,method=method,n=n,
            rmse=float(np.sqrt(np.mean([r['error']**2 for r in rr]))),worst=max(r['error'] for r in rr),
            allowance=float((max(r['error'] for r in rr)+u)/(1-u)),
            seconds=float(np.median([r['seconds'] for r in rr])),mineig=min(r['mineig'] for r in rr),
            psd=bool(min(r['mineig'] for r in rr)>=-1e-10),reference=ref))
    comparisons=[]; scientific={law:[] for law in ('gamma1','laplace')}
    for law in ('gamma1','laplace'):
        for d in (4,16,32,64):
            for geo in ('dense','sparse','near'):
                ss=[r for r in summary if (r['law'],r['d'],r['geometry'])==(law,d,geo)]
                base=next(r for r in ss if r['method']=='gaussian'); primary=next(r for r in ss if r['method']=='second_order')
                # The mechanism ratio is descriptive; reference validity is separately required.
                ratio=primary['rmse']/base['rmse']; ref=primary['reference']; u=ref['three_se']
                conservative_ratio=(primary['rmse']+u)/max(base['rmse']-u,1e-300)
                if geo=='dense' and d>=16 and conservative_ratio<=.8 and ref['convergence']: scientific[law].append(d)
                for eps in (.01,.003,.001):
                    valid=bool(u<=eps/10 and ref['convergence'])
                    eligible=[r for r in ss if r['method'] in ('gaussian','sobol','conditional','quadratic_cv') and r['allowance']<=eps and r['psd']]
                    winner=min(eligible,key=lambda r:r['seconds']) if eligible and valid else None
                    attained=bool(valid and primary['allowance']<=eps and primary['psd'])
                    tr=primary['seconds']/winner['seconds'] if winner is not None else None
                    comparisons.append(dict(law=law,d=d,geometry=geo,target=eps,reference_valid=valid,
                        gaussian_error=base['rmse'],primary_error=primary['rmse'],error_ratio=ratio,
                        conservative_error_ratio=conservative_ratio,primary_psd=primary['psd'],
                        primary_allowance=primary['allowance'],primary_seconds=primary['seconds'],
                        primary_attained=attained,competitor=winner['method'] if winner else None,
                        competitor_n=winner['n'] if winner else None,competitor_seconds=winner['seconds'] if winner else None,
                        time_ratio=tr,practical=bool(attained and tr is not None and tr<=.8)))
    wins=[(r['law'],r['d']) for r in comparisons if r['geometry']=='dense' and r['target']==.003 and r['d']>=16 and r['practical']]
    return dict(summary=summary,comparisons=comparisons,scientific_cells=scientific,
                scientific_gate=all(len(v)>=2 for v in scientific.values()),practical_cells=wins,
                practical_gate=len(wins)>=2,rows=len(rows))


def main():
    import sys
    refined='--refined' in sys.argv
    resolved='--resolved' in sys.argv
    dest=ROOT/'evidence/projected_resolution' if resolved else ROOT/'evidence/projected_refined' if refined else OUT
    result=analyze(refined,resolved); (dest/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in ('summary','comparisons'):
        with (dest/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(result[name][0]));w.writeheader();w.writerows(result[name])
    text=['# Двовимірна кумулянтна апроксимація ReLU-ядра: пілот',
          '\n2026-10-24. Окремий експеримент за PROJECTED_CUMULANT_SPEC.md.',
          f"\nScientific gate: **{'PASS' if result['scientific_gate'] else 'FAIL'}**; practical gate: **{'PASS' if result['practical_gate'] else 'FAIL'}**.",
          ('\nPost-primary завершення: еталон розширено до 8×2^23 для d32/d64; quadratic-CV Sobol додано як сильнішого конкурента для dense d>=16. Початкові оцінки й часи кандидата незмінні; конкурент виміряний в окремій сесії.' if resolved else '\nPost-primary уточнення еталона: точно центровані квадратичні контролі, окремі seeds і manifest. Початкові оцінки й часи незмінні.' if refined else '\nПервинний еталон; подальші уточнення наведено в окремих звітах.'),
          '\n## Що перевірено',
          '\nДва закони ваг, d=4/16/32/64, по8 входів у dense/sparse/near геометріях. '
          'ReLU без зміщення; перший шар. Gaussian, third, fourth і primary second_order, '
          'Sobol та conditional Sobol. Кандидат primary включає κ3, κ4 та κ3²/2. '
          'Час містить геометрію/кумулянти або генерацію вузлів; single BLAS thread, '
          'один warmup і5 вимірювань. Підбору truncation order за test немає.',
          '\n## Результати primary при target .003',
          '\n| Закон | d | Геометрія | Gaussian error | Primary error | Error ratio | PSD | Primary ms | Найшвидший конкурент | Time ratio | Статус |',
          '|---|---:|---|---:|---:|---:|---|---:|---|---:|---|']
    for r in result['comparisons']:
        if r['target']!=.003: continue
        status='REF-LIMIT' if not r['reference_valid'] else 'NOT-ATTAINED' if not r['primary_attained'] else 'WIN' if r['practical'] else 'ATTAINED'
        competitor=f"{r['competitor']} N={r['competitor_n']}" if r['competitor'] else '—'
        tr=f"{r['time_ratio']:.3f}" if r['time_ratio'] is not None and r['primary_attained'] else '—'
        text.append(f"| {r['law']} | {r['d']} | {r['geometry']} | {r['gaussian_error']:.6g} | {r['primary_error']:.6g} | {r['error_ratio']:.3f} | {r['primary_psd']} | {r['primary_seconds']*1000:.3f} | {competitor} | {tr} | {status} |")
    text+=['\n## Межі інтерпретації',
           '\nReference — незалежний scrambled QMC із coarse/fine перевіркою та8 реплікаціями; '
           '3SE — емпіричний показник, не доведена межа. Target допускається лише за '
           '3SE<=target/10 і проходження convergence. Conservative empirical allowance '
           '(observed error+3SE)/(1−3SE) використовується для всіх методів; для Sobol '
           'береться найгірша із8 помилок. Це описовий ретроспективний time-to-target, '
           'не гарантований stopping rule. Сітка N дискретна та різна за максимальним N '
           'для Sobol і quadratic-CV Sobol; speed ratio стосується лише перевіреної сітки. '
           'По одному fixed panel із8 входів на геометрію/d: немає оцінки переносу '
           'на інші datasets або великі kernel matrices. Час залежить від середовища; temporal drift '
           'та8 seeds не дають загальної гарантії прискорення.',
           '\nПоказник Error ratio не є гарантією для всіх входів. Sparse/near — окремі '
           'stress tests; мала κ у вихідних координатах не гарантує доброї whitened '
           'апроксимації. PSD repair і clipping не застосовувалися. Непроходження PSD '
           'не приховується навіть за малої Frobenius error.',
           '\nМетод є застосуванням Gaussian/Hermite/Edgeworth апарату. Цей пілот '
           'не встановлює новизни, користі PATP, теореми залишку для ReLU, '
           'переваги для глибокої CNN або після навчання. Негаусівський перший шар '
           'і finite-width correction прихованих шарів — різні задачі.',
           '\nПовні результати всіх порядків, targets і N: '+str(dest.relative_to(ROOT))+'/summary.csv '
           'та comparisons.csv. Сирі матриці, timings, references і hashes збережені.']
    (ROOT/('PROJECTED_RESOLVED_REPORT_UA.md' if resolved else 'PROJECTED_REFINED_REPORT_UA.md' if refined else 'PROJECTED_CUMULANT_REPORT_UA.md')).write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('summary','comparisons')},indent=2))

if __name__=='__main__': main()
