#!/usr/bin/env python3
"""Validate presence and hashes of external DFT/electronic analysis outputs."""
import argparse,csv,hashlib,json
from pathlib import Path
EXPECTED={'qe_relax':'relax.out','qe_scf':'scf.out','qe_nscf':'nscf.out','bader':'ACF.dat','elf':'elf.cube','cohp':'ICOHP.lobster','wannier':'wannier90.wout','epw':'epw.out'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
 for d in sorted(x for x in a.results_root.iterdir() if x.is_dir()):
  for stage,name in EXPECTED.items():
   f=d/name;exists=f.is_file();text=f.read_text(errors='ignore') if exists and name.endswith('.out') else ''
   converged=('convergence has been achieved' in text.lower()) if stage.startswith('qe_') and exists else None
   rows.append({'structure_id':d.name,'stage':stage,'status':'completed' if exists and converged is not False else ('not_converged' if exists else 'blocked'),'file':str(f) if exists else '', 'sha256':hashlib.sha256(f.read_bytes()).hexdigest() if exists else '', 'convergence':converged if converged is not None else 'not_applicable_or_unverified','software_version':'must be supplied in job manifest'})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['structure_id','stage','status','file','sha256','convergence','software_version']);w.writeheader();w.writerows(rows)
 print(json.dumps({'result_files_checked':len(rows),'completed':sum(r['status']=='completed' for r in rows),'note':'presence checks only; scientific review and full convergence validation remain required'},indent=2))
if __name__=='__main__':main()
