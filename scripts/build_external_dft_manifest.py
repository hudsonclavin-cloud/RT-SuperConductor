#!/usr/bin/env python3
"""Create blocked-by-default external DFT jobs with reproducible input hashes."""
import argparse,csv,hashlib,json
from pathlib import Path

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--cif-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pseudopotential-manifest',type=Path);a=p.parse_args()
 pseudos=set()
 if a.pseudopotential_manifest and a.pseudopotential_manifest.exists():
  pseudos={r['element'] for r in csv.DictReader(a.pseudopotential_manifest.open())}
 rows=[]
 for r in csv.DictReader(a.registry.open()):
  cif=a.cif_dir/r['cif_path']; elements=[]
  try:
   from pymatgen.core import Structure
   elements=sorted({e.symbol for e in Structure.from_file(cif).composition.elements})
  except Exception: pass
  missing=sorted(set(elements)-pseudos)
  status='queued' if elements and not missing and a.pseudopotential_manifest else 'blocked'
  rows.append({'structure_id':r['structure_id'],'mp_id':r['mp_id'],'cif_path':str(cif),'cif_sha256':digest(cif) if cif.exists() else '', 'elements':';'.join(elements),'missing_pseudopotentials':';'.join(missing),'status':status,'software':'Quantum ESPRESSO; version pending','functional':'pending','ecutwfc_Ry':'pending','kmesh':'pending','commands':'pending configuration','expected_outputs':'relax.out;scf.out;nscf.out;ph.out','convergence':'not_run','failure_reason':'pseudopotential manifest/settings missing' if status=='blocked' else ''})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['structure_id','status']);w.writeheader();w.writerows(rows)
 print(json.dumps({'jobs':len(rows),'queued':sum(r['status']=='queued' for r in rows),'blocked':sum(r['status']=='blocked' for r in rows),'reason':'No DFT jobs run by this manifest builder.'},indent=2))
if __name__=='__main__':main()
