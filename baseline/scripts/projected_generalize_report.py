"""Recompute all panel decisions from stored matrices; verify and report."""
from projected_generalize import *

def main():
    checks={}
    for name,sha in json.loads((OUT/'manifest.json').read_text()).items():
        checks['hash:'+name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    with (OUT/'raw.csv').open() as f:rows=list(csv.DictReader(f))
    kernels=np.load(OUT/'kernels.npz');ts=json.loads((OUT/'timings.json').read_text());ds=json.loads((OUT/'diagnostics.json').read_text())
    checks['row_count']=len(rows)==18*66
    summary=[]
    for li,law in enumerate(('gamma1','laplace')):
        for d in (16,64,128):
            for panel in range(3):
                cell=f'{law}_d{d}_p{panel}'; data=np.load(OUT/f'reference_{cell}.npz');x=data['inputs'];ref=data['mean'];norm=np.linalg.norm(ref)
                checks[cell+':inputs']=np.array_equal(x,inputs(d,panel)) and np.allclose(np.linalg.norm(x,axis=1),1)
                level=max(int(k[5:]) for k in data if k.startswith('level'))
                diag=diagnostics(data[f'level{level}'],data[f'level{level-1}'],data[f'direct{level}'])
                checks[cell+':diag']=diag==ds[cell]
                checks[cell+':mean']=np.array_equal(ref,data[f'level{level}'].mean(0))
                groups={}
                for row in [r for r in rows if r['cell']==cell]:
                    k=kernels[row['key']];err=float(np.linalg.norm(k-ref)/norm);eig=float(np.linalg.eigvalsh(k)[0]/norm)
                    checks[row['key']+':metrics']=abs(err-float(row['error']))<1e-14 and abs(eig-float(row['mineig']))<1e-14 and float(row['seconds'])==float(np.median(ts[row['key']]))
                    if row['method'] in ('gaussian','second_order'):
                        order=0 if row['method']=='gaussian' else 3
                        checks[row['key']+':replay']=np.allclose(k,matrices(x,law,order)[order],rtol=0,atol=1e-14)
                    elif int(row['n'])==4096 and int(row['rep'])==0:
                        seed=30000000+li*1000000+d*1000+panel*100
                        fresh=sobol_kernel(x[None],law,4096,seed)[0][0] if row['method']=='sobol' else replicate(x[None],law,4096,seed)[0][0]
                        checks[row['key']+':replay']=np.allclose(k,fresh,rtol=0,atol=1e-14)
                    groups.setdefault((row['method'],int(row['n'])),[]).append(row)
                stats=[]
                for (method,n),rr in groups.items():
                    bound=(max(float(r['error']) for r in rr)+diag['u'])/(1-diag['u'])
                    psd=min(float(r['mineig']) for r in rr)>=-1e-10
                    stats.append(dict(method=method,n=n,bound=bound,error=max(float(r['error']) for r in rr),psd=psd,seconds=float(np.median([float(r['seconds']) for r in rr])),eligible=diag['valid'] and bound<=.003 and psd))
                candidate=next(s for s in stats if s['method']=='second_order');gauss=next(s for s in stats if s['method']=='gaussian')
                competitors=[s for s in stats if s['method']!='second_order' and s['eligible']]
                best=min(competitors,key=lambda s:s['seconds']) if competitors else None
                ratio=candidate['seconds']/best['seconds'] if best else None
                status='REFERENCE_LIMITED' if not diag['valid'] else 'CANDIDATE_FAIL' if not candidate['eligible'] else 'COMPETITOR_LIMITED' if best is None else 'PASS' if ratio<=.8 else 'NO_SPEED_GAIN'
                summary.append(dict(cell=cell,law=law,d=d,panel=panel,u=diag['u'],status=status,candidate=candidate,gaussian=gauss,best=best,ratio=ratio,methods=stats))
    checks['distinct_input_seeds']=len({2026102500+100*d+p for d in (16,64,128) for p in range(3)})==9
    reference_seeds={10000000+li*1000000+d*1000+p*100+level*10+r for li in range(2) for d in (16,64,128) for p in range(3) for level in range(3) for r in range(8)}
    method_seeds={30000000+li*1000000+d*1000+p*100+r for li in range(2) for d in (16,64,128) for p in range(3) for r in range(8)}
    checks['seed_independence']=len(reference_seeds)==432 and len(method_seeds)==144 and reference_seeds.isdisjoint(method_seeds)
    checks['positive_finite_timings']=all(len(v)==5 and np.all(np.isfinite(v)) and min(v)>0 for v in ts.values())
    result=dict(passed=sum(checks.values()),total=len(checks),checks=checks)
    dump(OUT/'verification.json',result)
    if not all(checks.values()):raise RuntimeError([k for k,v in checks.items() if not v])
    selected=[s for s in summary if (s['law'],s['d']) in (('gamma1',64),('laplace',16))]
    gate=all(s['status']=='PASS' for s in selected)
    dump(OUT/'analysis.json',dict(transfer_gate=gate,cells=summary))
    lines=['# Незалежні панелі та матриці 32×32','',f'Гейт перенесення на всі шість наперед визначених панелей: **{"PASS" if gate else "NOT PASSED"}**.','',
        'Три незалежні панелі на розмірність; d=16,64,128; два відомі iid закони. Формула second_order та всі попередні свідчення незмінні. Похибка — відносна норма Фробеніуса повної матриці. Час — повне обчислення матриці, один BLAS thread, median 5 repeats після warmup. Калібрування відсутнє.','',
        '| Закон, d, панель | Gaussian error | Correction error | Correction allowance | Reference 3SE | Статус | Час correction, ms | Найшвидший придатний конкурент | Відношення часу |',
        '|---|---:|---:|---:|---:|---|---:|---|---:|']
    for s in summary:
        c=s['candidate'];b=s['best'];label=f"{b['method']} N={b['n']} ({b['seconds']*1e3:.3f} ms)" if b else '—'
        ratio=f"{s['ratio']:.4f}" if s['ratio'] is not None and c['eligible'] else '—'
        lines.append(f"| {s['cell']} | {s['gaussian']['error']:.6g} | {c['error']:.6g} | {c['bound']:.6g} | {s['u']:.6g} | {s['status']} | {c['seconds']*1e3:.3f} | {label} | {ratio} |")
    lines+=['','Статуси: PASS — точність .003 і PSD пройдено, час ≤80% найшвидшого придатного конкурента. CANDIDATE_FAIL — кандидат не досяг точності або PSD. NO_SPEED_GAIN — точність є, виграшу часу немає. REFERENCE_LIMITED — еталон не дозволяє рішення. COMPETITOR_LIMITED — сітка конкурентів не дає скінченного відношення часу.','',
        'Порівняння обмежене Gaussian, direct Sobol та quadratic-CV Sobol на N=4096,16384,65536,262144. Conditional Sobol тут не вимірювався. Великі коефіцієнти прискорення не є твердженням про оптимальний можливий алгоритм. 3SE та allowance — емпіричні діагностики, не доведена ймовірнісна межа.','',
        'Три панелі не встановлюють універсальну надійність; елементи матриці не трактуються як незалежні повторення. Sparse/near обмеження попереднього пілота залишаються чинними. Немає висновку про PATP, невідомі кумулянти, навчання CNN або новизну методу.','',
        f"Технічна перевірка: {result['passed']}/{result['total']}. Перераховано всі метрики, replay всіх deterministic матриць та N4096 rep0 обох стохастичних методів у кожній клітині; перевірено SHA256, inputs, reference means/diagnostics. Повний еталонний прогін повторно не виконувався.",'',
        'Протокол: [PROJECTED_GENERALIZATION_SPEC.md](PROJECTED_GENERALIZATION_SPEC.md). Повні дані: [analysis.json](evidence/projected_generalization/analysis.json), [verification.json](evidence/projected_generalization/verification.json).']
    (ROOT/'PROJECTED_GENERALIZATION_REPORT_UA.md').write_text('\n'.join(lines)+'\n')
    files=[Path(__file__),ROOT/'PROJECTED_GENERALIZATION_REPORT_UA.md']+[OUT/n for n in ('verification.json','analysis.json')]
    dump(OUT/'analysis_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print('VERIFIED',result['passed'],result['total'],'TRANSFER',gate)
    for s in summary:print(s['cell'],s['status'],s['candidate']['bound'],s['ratio'])
if __name__=='__main__':main()
