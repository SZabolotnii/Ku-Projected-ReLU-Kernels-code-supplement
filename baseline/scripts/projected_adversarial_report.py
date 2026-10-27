"""Audit immutable inputs and summarize the new same-session comparison."""
from projected_adversarial import *

def main():
    checks={}
    for folder in (OLD,REF,OUT):
        for mf in ('manifest.json','analysis_manifest.json') if folder!=OUT else ('manifest.json',):
            for name,sha in json.loads((folder/mf).read_text()).items():checks['hash:'+name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    pre=json.loads((OUT/'preflight.json').read_text());checks.update({'preflight:'+k:v for k,v in pre['checks'].items()})
    rows=list(csv.DictReader((OUT/'raw.csv').open()));ks=np.load(OUT/'kernels.npz');oldks=np.load(OLD/'kernels.npz');ts=json.loads((OUT/'timings.json').read_text());result=[]
    checks['rows']=len(rows)==348
    for li,(law,d) in enumerate((('gamma1',64),('laplace',16))):
        for panel in range(3):
            cell=f'{law}_d{d}_p{panel}';rp=REF if law=='gamma1' else OLD
            ref=np.load(rp/f'reference_{cell}.npz')['mean'];diag=json.loads((rp/'diagnostics.json').read_text())[cell];u=diag['u'];norm=np.linalg.norm(ref);groups={}
            for r in [r for r in rows if r['cell']==cell]:
                k=ks[r['key']];error=float(np.linalg.norm(k-ref)/norm);eig=float(np.linalg.eigvalsh(k)[0]/norm)
                checks[r['key']+':finite_symmetric']=bool(np.all(np.isfinite(k)) and np.allclose(k,k.T,atol=1e-14,rtol=0))
                checks[r['key']+':time']=len(ts[r['key']])==5 and min(ts[r['key']])>0 and float(r['seconds'])==float(np.median(ts[r['key']]))
                if r['key'] in oldks:checks[r['key']+':frozen_parity']=np.allclose(k,oldks[r['key']],rtol=0,atol=1e-14)
                if int(r['rep'])==0 and r['method']=='conditional':
                    fresh=conditional_kernel(inputs(d,panel),law,int(r['n']),int(r['seed']))
                    checks[r['key']+':replay']=np.allclose(k,fresh,atol=1e-14,rtol=0)
                groups.setdefault((r['method'],int(r['n'])),[]).append(dict(error=error,eig=eig,seconds=float(r['seconds'])))
            stats=[]
            for (method,n),rr in groups.items():
                error=max(r['error'] for r in rr);bound=(error+u)/(1-u);psd=min(r['eig'] for r in rr)>=-1e-10
                stats.append(dict(method=method,n=n,error=error,bound=bound,psd=psd,seconds=float(np.median([r['seconds'] for r in rr])),eligible=bool(diag['valid'] and bound<=.003 and psd)))
            c=next(s for s in stats if s['method']=='second_order');bs=[s for s in stats if s['method']!='second_order' and s['eligible']];best=min(bs,key=lambda s:s['seconds']) if bs else None
            speed=best['seconds']/c['seconds'] if best else None
            passed=bool(c['eligible'] and speed is not None and speed>=1.25)
            result.append(dict(cell=cell,candidate=c,best=best,speedup=speed,passed=passed,methods=stats))
    verification=dict(passed=sum(checks.values()),total=len(checks),checks=checks);dump(OUT/'verification.json',verification)
    if not all(checks.values()):raise RuntimeError([k for k,v in checks.items() if not v])
    dump(OUT/'analysis.json',dict(all_pass=all(r['passed'] for r in result),cells=result))
    lines=['# Аудит сітки бюджетів і практичного прискорення','',f"Усі шість панелей проходять уточнене порівняння: **{all(r['passed'] for r in result)}**.",'',
      'Нові N=32768,131072 для Sobol та quadratic-CV; conditional Sobol N=1024,4096. Кандидат, Gaussian та попередній переможець переміряні в тій самій сесії. Еталони й формула незмінні.','',
      '| Панель | Allowance | Candidate ms | Найшвидший придатний конкурент | Конкурент ms | Прискорення |',
      '|---|---:|---:|---|---:|---:|']
    for r in result:
        c=r['candidate'];b=r['best'];lines.append(f"| {r['cell']} | {c['bound']:.7f} | {c['seconds']*1000:.3f} | {b['method'] if b else 'none'} N={b['n'] if b else 0} | {b['seconds']*1000 if b else float('nan'):.3f} | {r['speedup'] if b else float('nan'):.2f}× |")
    lines+=['','## Обмеження та перевірка формули','', 'Час помітно змінюється між сесіями: новий Gamma speedup 69–105× проти попередніх 67.5–70.7×; Laplace 17.7–23.9× проти 20.7–21.1×. Це не додатковий алгоритмічний виграш. Надійний висновок — збереження практичного критерію, а не стабільність точного множника.','',
      'Попередні 67.5–70.7× для Gamma стосувалися грубої сітки; нова таблиця дає актуальне вимірювання на доповненій сітці. Це не оптимум серед усіх можливих методів чи бюджетів. Conditional контроль обмежений N≤4096; його оптимальний час до .003 не встановлено. Старі невиміряні заново бюджети не змішуються з новими часами.','',
      'Для істинного K, еталона R та апроксимації A: за умови ||R−K||≤u||R|| трикутна нерівність дає ||A−K||/||K||≤(e+u)/(1−u), e=||A−R||/||R||. Знаменник коректний. Однак u=3 empirical SE не є доведеною confidence bound; твердження про гарантовану точність було б надмірним.','',
      'Повторний preflight порівнює аналітичні Hermite contractions з незалежною числовою квадратурою, перевіряє симетрії та граничні випадки. Це числова перевірка реалізації; вона не доводить залишковий член Edgeworth для ReLU і не забезпечує PSD для довільних входів.','',
      f"Технічні перевірки: {verification['passed']}/{verification['total']}; збережено {len(rows)} матриць і п’ять timing repeats для кожної. Перевірено frozen matrix parity та conditional replay. Повний reference run не повторювався.",'',
      'Known laws, dense inputs, три панелі на закон/розмірність, перший шар без зміщення. Новизна, PATP-перевага, невідомі кумулянти й навчання CNN залишаються неперевіреними.','',
      '[Протокол](PROJECTED_ADVERSARIAL_SPEC.md); [усі методи й бюджети](evidence/projected_adversarial/analysis.json); [верифікація](evidence/projected_adversarial/verification.json).']
    (ROOT/'PROJECTED_ADVERSARIAL_REPORT_UA.md').write_text('\n'.join(lines)+'\n')
    files=[Path(__file__),ROOT/'PROJECTED_ADVERSARIAL_REPORT_UA.md',OUT/'analysis.json',OUT/'verification.json']
    dump(OUT/'analysis_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print('VERIFIED',verification['passed'],verification['total'])
    for r in result:print(r['cell'],r['passed'],r['speedup'],r['best'])
if __name__=='__main__':main()
