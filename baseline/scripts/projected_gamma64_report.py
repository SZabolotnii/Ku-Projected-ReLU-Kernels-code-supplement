"""Verify references and reevaluate frozen candidate/comparator matrices."""
from projected_gamma64_refine import *

def main():
    checks={}
    for folder in (OLD,OUT):
        for manifest in ('manifest.json','analysis_manifest.json') if folder==OLD else ('manifest.json',):
            for name,sha in json.loads((folder/manifest).read_text()).items():
                checks['hash:'+name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    checks['prior_verification']=all(json.loads((OLD/'verification.json').read_text())['checks'].values())
    rs=list(csv.DictReader((OLD/'raw.csv').open()));ks=np.load(OLD/'kernels.npz');timings=json.loads((OLD/'timings.json').read_text());ds=json.loads((OUT/'diagnostics.json').read_text());summary=[]
    for panel in range(3):
        cell=f'gamma1_d64_p{panel}';data=np.load(OUT/f'reference_{cell}.npz');previous=np.load(OLD/f'reference_{cell}.npz');level=max(int(k[5:]) for k in data if k.startswith('level'))
        coarse=previous['level2'] if level==0 else data['level0']
        fine=data[f'level{level}'];diag=diagnostics(fine,coarse,data[f'direct{level}']);ref=fine.mean(0);norm=np.linalg.norm(ref)
        checks[cell+':diag']=diag==ds[cell];checks[cell+':mean']=np.array_equal(ref,data['mean']);checks[cell+':inputs']=np.array_equal(data['inputs'],inputs(64,panel))
        checks[cell+':finite']=all(np.all(np.isfinite(data[k])) for k in data)
        groups={}
        for r in [r for r in rs if r['cell']==cell]:
            k=ks[r['key']];error=float(np.linalg.norm(k-ref)/norm);mineig=float(np.linalg.eigvalsh(k)[0]/norm)
            checks[r['key']+':time']=float(r['seconds'])==float(np.median(timings[r['key']]))
            checks[r['key']+':symmetry']=np.allclose(k,k.T,rtol=0,atol=1e-14)
            if r['method'] in ('gaussian','second_order'):
                order=0 if r['method']=='gaussian' else 3
                checks[r['key']+':replay']=np.allclose(k,matrices(data['inputs'],'gamma1',order)[order],rtol=0,atol=1e-14)
            groups.setdefault((r['method'],int(r['n'])),[]).append(dict(error=error,mineig=mineig,seconds=float(r['seconds'])))
        methods=[]
        for (method,n),rr in groups.items():
            error=max(r['error'] for r in rr);bound=(error+diag['u'])/(1-diag['u']);psd=min(r['mineig'] for r in rr)>=-1e-10
            methods.append(dict(method=method,n=n,error=error,bound=bound,psd=psd,seconds=float(np.median([r['seconds'] for r in rr])),eligible=bool(diag['valid'] and bound<=.003 and psd)))
        c=next(m for m in methods if m['method']=='second_order');g=next(m for m in methods if m['method']=='gaussian');eligible=[m for m in methods if m['method']!='second_order' and m['eligible']];best=min(eligible,key=lambda m:m['seconds']) if eligible else None
        ratio=c['seconds']/best['seconds'] if best else None
        status='REFERENCE_LIMITED' if not diag['valid'] else 'CANDIDATE_FAIL' if not c['eligible'] else 'COMPETITOR_LIMITED' if best is None else 'PASS' if ratio<=.8 else 'NO_SPEED_GAIN'
        summary.append(dict(cell=cell,diagnostics=diag,candidate=c,gaussian=g,best=best,ratio=ratio,status=status,methods=methods))
    checks['row_count']=sum(r['law']=='gamma1' and r['d']=='64' for r in rs)==198
    report=dict(checks=checks,passed=sum(checks.values()),total=len(checks));dump(OUT/'verification.json',report)
    if not all(checks.values()):raise RuntimeError([k for k,v in checks.items() if not v])
    old=json.loads((OLD/'analysis.json').read_text());lap=[s for s in old['cells'] if s['law']=='laplace' and s['d']==16]
    gate=all(s['status']=='PASS' for s in summary+lap)
    dump(OUT/'analysis.json',dict(post_primary_transfer_gate=gate,gamma_cells=summary,inherited_laplace_cells=lap))
    lines=['# Gamma d64: уточнений еталон незалежних панелей','',f'Post-primary гейт перенесення Gamma64 + успадковані Laplace16: **{"PASS" if gate else "NOT PASSED"}**.','',
      'Три незалежні панелі по 32 входи; відомий standardized Gamma(1), d=64, unshifted ReLU. Незмінні candidate/comparator матриці та timings з generalization; лише еталон та похибки уточнено. Первинний NOT PASSED зберігається як історичний результат.','',
      '| Панель | Reference 3SE | Gaussian error | Correction error | Allowance | Час, ms | Конкурент | Прискорення | Статус |',
      '|---|---:|---:|---:|---:|---:|---|---:|---|']
    for s in summary:
        c=s['candidate'];b=s['best'];label=f"{b['method']} N={b['n']}, {b['seconds']*1000:.3f} ms" if b else '—';speed=f"{1/s['ratio']:.2f}×" if s['ratio'] is not None and c['eligible'] else '—'
        lines.append(f"| {s['cell']} | {s['diagnostics']['u']:.8f} | {s['gaussian']['error']:.7f} | {c['error']:.7f} | {c['bound']:.7f} | {c['seconds']*1000:.3f} | {label} | {speed} | {s['status']} |")
    lines+=['','Критерії: reference 3SE≤.0003 і coarse/fine convergence; allowance≤.003; PSD без repair; час ≤80% найшвидшого придатного конкурента. Якщо конкурент не досяг цілі на N≤262144, статус COMPETITOR_LIMITED без кількісного speedup.','',
      'Відношення часу стосується фіксованої сітки Gaussian/direct Sobol/quadratic-CV Sobol. Conditional Sobol не вимірювався. Вартість еталона не входить у runtime методу. Час узятий з попередньої сесії одночасного порівняння методів. Калібрування відсутнє.','',
      'Це post-primary reference refinement, а не новий confirmatory дизайн. Емпірична 3SE не є доведеною межею ймовірності. Три панелі не встановлюють універсальної переваги; Gamma16, sparse/near та невизначені d128 залишаються поза позитивним висновком. Новизна, PATP, навчання CNN та невідомі закони не перевірялися.','',
      f"Технічна перевірка {report['passed']}/{report['total']}: hashes, усі 198 переоцінених матриць, timings, inputs, reference means/diagnostics, deterministic replay. Повний еталон вдруге не запускався.",'',
      '[Протокол](PROJECTED_GAMMA64_REFINEMENT_SPEC.md), [повні результати](evidence/projected_gamma64_refinement/analysis.json), [верифікація](evidence/projected_gamma64_refinement/verification.json).']
    (ROOT/'PROJECTED_GAMMA64_REPORT_UA.md').write_text('\n'.join(lines)+'\n')
    files=[Path(__file__),ROOT/'PROJECTED_GAMMA64_REPORT_UA.md',OUT/'verification.json',OUT/'analysis.json']
    dump(OUT/'analysis_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print(report['passed'],report['total'],'TRANSFER',gate)
    for s in summary:print(s['cell'],s['status'],s['candidate']['bound'],s['ratio'])
if __name__=='__main__':main()
