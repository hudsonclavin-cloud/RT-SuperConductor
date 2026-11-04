#!/usr/bin/env python3
"""Secret-backed Actions-only MP elasticity and DOS enrichment with missingness."""
import argparse,csv,json,os,sys,re
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rt_superconductor.harmonized import load_harmonized

def safe_error(error):
    message=type(error).__name__+': '+str(error)
    key=os.environ.get('MP_API_KEY')
    return (message.replace(key,'[redacted]') if key else message)[:700]

def dos_at_fermi(dos):
    energies=np.asarray(dos.energies,dtype=float);density=np.asarray(dos.get_densities(),dtype=float);ef=float(dos.efermi)
    if energies.ndim!=1 or density.shape!=energies.shape or len(energies)<2 or not np.isfinite(energies).all() or not np.isfinite(density).all() or not np.isfinite(ef) or np.any(np.diff(energies)<=0) or not energies[0]<=ef<=energies[-1]:
        raise ValueError('Invalid DOS energy grid, density, or Fermi-level coverage')
    value=float(np.interp(ef,energies,density))
    if value<0:raise ValueError('Negative total DOS is not a physical observation')
    return value
def orbital_weights(dos):
    energies=np.asarray(dos.energies);ef=float(dos.efermi);lo,hi=ef-.5,ef+.5
    if energies.min()>lo or energies.max()<hi:return {}
    grid=np.r_[lo,energies[(energies>lo)&(energies<hi)],hi];weights={};elements={}
    for element in dos.structure.composition.elements:
        es={}
        for orbital,projection in dos.get_element_spd_dos(element).items():
            value=float(trapezoid(np.interp(grid,energies,projection.get_densities()),grid));key=str(orbital);es[key]=value;weights[key]=weights.get(key,0)+value
        elements[str(element)]=es
    total=sum(weights.values())
    if not np.isfinite(total) or total<=0 or any(not np.isfinite(v) or v<0 for v in weights.values()):return {}
    return {**{f'orbital_{l}_fraction':weights.get(l,0)/total for l in ('s','p','d','f')},'element_orbital_integrals_json':json.dumps(elements),'orbital_window_ev':1.0}

def dos_storage_keys(task_id,normalize=None):
    if normalize is None:
        from mp_api.client.core.utils import validate_ids
        normalize=lambda task:validate_ids([task])[0]
    normalized=str(normalize(task_id))
    # The new identifier validator strips the old task prefix; S3 may retain it.
    legacy='mp-'+normalized if normalized.isdigit() else normalized
    return list(dict.fromkeys('dos/'+key+'.json.gz' for key in (legacy,str(task_id))))

def dos_curve_for_task(route,task_id):
    try:return route.get_dos_from_task_id(task_id)
    except OSError as original:
        if not any(token in str(original) for token in ('NoSuchKey','specified key does not exist')):raise
        for key in dos_storage_keys(task_id):
            try:
                result,_=route._query_open_data(bucket='materialsproject-parsed',key=key)
                if result:return result[0]['data']
            except OSError as error:
                if not any(token in str(error) for token in ('NoSuchKey','specified key does not exist')):raise
        raise original

def complete_dos_with_task(client,material_id):
    docs=client.materials.electronic_structure.search(material_ids=[material_id],fields=['dos'])
    if not docs or not docs[0].get('dos'):return None,None
    metadata=docs[0]['dos'];total=metadata.get('total',{});spin=total.get('1',{});task=metadata.get('task_id') or spin.get('task_id')
    if not task:raise ValueError('DOS metadata exists but lacks the required calculation task ID; metadata='+repr(docs[0]['dos'])[:1500])
    task_id=str(task);return dos_curve_for_task(client.materials.electronic_structure_dos,task_id),task_id

def flush_output(rows,by_id,cache,output,report):
    columns=('mp_bulk_modulus_gpa','mp_dos_ef_states_ev_cell','orbital_s_fraction','orbital_p_fraction','orbital_d_fraction','orbital_f_fraction','element_orbital_integrals_json','orbital_window_ev','mp_dos_task_id','mp_dos_content_sha256','mp_orbital_error','mp_bulk_error','mp_dos_error')
    for r in rows:
        mid=by_id.get(r['record_id']);d=cache.get(mid,{})
        r.update({k:d.get(k) if d.get(k) is not None else r.get(k) for k in columns});r.update(reference_mp_id=mid,mp_bulk_status=d.get('bulk_status','not_queried'),mp_dos_status=d.get('dos_status','not_queried'),mp_orbital_status=d.get('orbital_status','not_queried'))
    report.update(bulk_modulus_available=sum(r.get('mp_bulk_modulus_gpa') not in (None,'') for r in rows),dos_available=sum(r.get('mp_dos_ef_states_ev_cell') not in (None,'') for r in rows),orbital_available=sum(r.get('orbital_d_fraction') not in (None,'') for r in rows))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(report);return report

def select_reference_ids(values,requested=None):
    available=sorted({v for v in values if isinstance(v,str) and re.fullmatch(r'(?:mp|mvc)-[0-9]+',v)})
    if requested is not None:
        missing=set(requested)-set(available)
        if missing:raise ValueError('Requested reference IDs are not in this cohort: '+str(sorted(missing)))
        return [v for v in available if v in requested]
    return available

def enrich(input_path,features,output,limit=100,requested_ids=None):
    import hashlib,importlib.metadata
    if limit<0:raise ValueError('limit must be non-negative; zero means all reference IDs')
    rows=list(csv.DictReader(features.open()));by_id={r.record_id:r.mp_id for r in load_harmonized(input_path)}
    report={'status':'blocked','reason':'requires GitHub Actions and MP_API_KEY','queries':0,'bulk_modulus_available':0,'dos_available':0,'orbital_available':0,'failed_endpoint_calls':0,'endpoint_error_samples':[],'input_rows':len(rows),'feature_input_sha256':hashlib.sha256(features.read_bytes()).hexdigest(),'harmonized_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),'source':'https://api.materialsproject.org','orbital_window_relative_ef_ev':[-.5,.5],'orbital_normalization':'sum of all available element/orbital projected integrals including f, not total-DOS or electron occupations','units':{'bulk':'GPa','dos_ef':'states/eV/cell','orbital_fractions':'dimensionless'},'provenance':'DFT-computed reference structures, not measured sample properties','mp_document_models':False,'dos_hash_method':'SHA-256 of sorted compact Monty JSON of the CompleteDos object used for feature extraction','numeric_validation':'finite property values and normalized orbital weights; missing/invalid values remain missing'}
    cache={};cache_dir=output.parent/'mp-property-cache'
    report['cache_hits']=0
    if os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('MP_API_KEY'):
        from mp_api.client import MPRester
        import time
        cache_dir.mkdir(parents=True,exist_ok=True)
        report['mp_api_version']=importlib.metadata.version('mp-api');report.update(status='collecting',reason='Partial snapshot while reference-property queries are active')
        all_ids=select_reference_ids(by_id.get(r['record_id']) for r in rows);report['unique_reference_ids']=len(all_ids);ids=select_reference_ids(all_ids,requested_ids);report['selected_reference_ids']=len(ids);report['requested_reference_ids']=sorted(requested_ids) if requested_ids is not None else None;ids=ids[:limit] if limit else ids;streak=0
        try:
            with MPRester(os.environ['MP_API_KEY'],use_document_model=False) as client:
                for mid in ids:
                    cache_file=cache_dir/(mid+'.json')
                    if cache_file.exists() and time.time()-cache_file.stat().st_mtime<86400:
                        cached=json.loads(cache_file.read_text())
                        if cached.get('collector_version')==4 and cached.get('mp_api_version')==report['mp_api_version']:
                            cache[mid]=cached['data'];report['cache_hits']+=1;continue
                    d={'mp_bulk_modulus_gpa':None,'mp_dos_ef_states_ev_cell':None};report['queries']+=1;auth_failures=0
                    try:
                        docs=client.materials.elasticity.search(material_ids=[mid],fields=['material_id','bulk_modulus'])
                        if docs:
                            k=docs[0].get('bulk_modulus') if isinstance(docs[0],dict) else docs[0].bulk_modulus
                            if k is not None:d['mp_bulk_modulus_gpa']=getattr(k,'vrh',None) if not isinstance(k,dict) else k.get('vrh')
                        if d['mp_bulk_modulus_gpa'] is not None and not np.isfinite(float(d['mp_bulk_modulus_gpa'])):d['mp_bulk_modulus_gpa']=None
                        d['bulk_status']='available' if d['mp_bulk_modulus_gpa'] is not None else 'not_in_reference'
                    except Exception as e:
                        auth=any(code in str(e) for code in ('401','403'));auth_failures+=auth;d['bulk_status']='blocked_authorization' if auth else 'blocked_'+type(e).__name__;d['mp_bulk_error']=safe_error(e);report['failed_endpoint_calls']+=1
                    try:
                        dos,task_id=complete_dos_with_task(client,mid);d['mp_dos_task_id']=task_id
                        if dos is not None:
                            from monty.json import MontyEncoder
                            d['mp_dos_content_sha256']=hashlib.sha256(json.dumps(dos.as_dict(),cls=MontyEncoder,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                            value=dos_at_fermi(dos);d['mp_dos_ef_states_ev_cell']=value if np.isfinite(value) else None
                            try:d.update(orbital_weights(dos));d['orbital_status']='available' if 'orbital_d_fraction' in d else 'missing_projections_or_window'
                            except Exception as e:
                                d['orbital_status']='blocked_'+type(e).__name__;d['mp_orbital_error']=safe_error(e)
                                if len(report['endpoint_error_samples'])<3:report['endpoint_error_samples'].append({'material_id':mid,'endpoint':'orbital_extraction','error':safe_error(e)})
                        d['dos_status']='available' if dos is not None else 'not_in_reference'
                        if dos is None:d['orbital_status']='not_in_reference_dos'
                    except Exception as e:
                        auth=any(code in str(e) for code in ('401','403'));auth_failures+=auth;d['dos_status']='blocked_authorization' if auth else 'blocked_'+type(e).__name__;d['mp_dos_error']=safe_error(e);report['failed_endpoint_calls']+=1
                        if len(report['endpoint_error_samples'])<3:report['endpoint_error_samples'].append({'material_id':mid,'endpoint':'dos','error':safe_error(e)})
                    cache[mid]=d
                    if not any(str(d.get(k,'')).startswith('blocked') for k in ('bulk_status','dos_status','orbital_status')):
                        cache_file.write_text(json.dumps({'collector_version':4,'mp_api_version':report['mp_api_version'],'data':d},allow_nan=False))
                    if report['queries']%50==0:flush_output(rows,by_id,cache,output,report)
                    streak=streak+1 if auth_failures==2 else 0
                    if streak>=3:
                        report.update(status='blocked_api_authorization',reason='Three successive reference IDs rejected both property endpoints with HTTP 401/403; remaining IDs unqueried');break
                else:report.update(status='completed_with_missingness',reason='Missing properties and projections remain missing')
        except Exception as e:report.update(status='blocked_api_client',reason=type(e).__name__)
    return flush_output(rows,by_id,cache,output,report)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('data/training/harmonized_supercon.csv'));p.add_argument('--features',type=Path,default=Path('data/processed/structure_features.csv'));p.add_argument('--output',type=Path,default=Path('data/processed/structure_features_mp.csv'));p.add_argument('--limit',type=int,default=100);p.add_argument('--ids',default='',help='Optional comma-separated reference MP/MVC IDs');a=p.parse_args();requested=set(a.ids.replace(',',' ').split()) or None;enrich(a.input,a.features,a.output,a.limit,requested)
