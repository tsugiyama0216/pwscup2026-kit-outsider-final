"""Joint synthetic world with public private-core demographics and release profiles.

No attack module, prediction or official feedback is imported/read here. The
protocol and seed list are fixed in generation_v4_protocol.json. Selection sees
only generated C, public challenge/background and actual public profile metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank_validation.public_profiles import (
    profile_tables, build_coverage_report, verify_original_inputs,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL_PATH = HERE / 'generation_v4_protocol.json'
PROTOCOL_SHA256 = 'c383c5fe06136fdbdd11d501467c7c4731919137cf2c8b7b8d701a524dc95038'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol():
    if _sha(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError('Generation protocol changed; create a new explicitly versioned protocol')
    return json.loads(PROTOCOL_PATH.read_text(encoding='utf-8-sig'))


def _reserved_split(seed, protocol):
    if seed in protocol['development_seeds']:
        return 'dev', 'development'
    if seed in protocol['confirmation_seeds']:
        return 'holdout', 'confirmation'
    raise ValueError('Seed is not reserved by the fixed generation protocol')


def public_targets(root, protocol):
    root = Path(root).resolve()
    attack = root / 'main_round/attack'
    bg_path = root / 'main_round/anonymization/team10/participant_data/A_bg.csv'
    background = pd.read_csv(bg_path, float_precision='round_trip')
    records, hashes = {}, {bg_path.relative_to(root).as_posix(): _sha(bg_path)}
    for path in sorted((attack / 'targets').glob('C_*.csv'), key=lambda p: int(p.stem[2:])):
        team = int(path.stem[2:])
        if team == 10:
            continue
        challenge_path = path.with_name(f'aia_challenge_{team}.csv')
        c = pd.read_csv(path, float_precision='round_trip')
        challenge = pd.read_csv(challenge_path, float_precision='round_trip')
        records[team] = {'name': f'C_{team}', 'cohort': team, **profile_tables(c, challenge, background)}
        for source in (path, challenge_path):
            hashes[source.relative_to(root).as_posix()] = _sha(source)
    scales = {}
    for metric in protocol['profile_metrics']:
        values = np.array([row[metric] for row in records.values()], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Invalid actual public profile: {metric}')
        iqr = float(np.quantile(values, .75) - np.quantile(values, .25))
        scales[metric] = max(iqr, protocol['scale_floors'][metric])
    return records, scales, hashes


def public_fit_loss(profile, desired, scales, protocol):
    loss, residuals = 0.0, {}
    for metric in protocol['profile_metrics']:
        value = profile.get(metric)
        if value is None or not np.isfinite(value):
            return float('inf'), {}
        z = float((value - desired[metric]) / scales[metric])
        residuals[metric] = z
        az = abs(z)
        term = z*z if az <= 2 else 4*az-4
        loss += protocol['metric_weights'][metric] * term
    return float(loss), residuals


def _censor(frame):
    frame = frame.copy(deep=True)
    frame.loc[(frame.onset == 0) & (frame.death == 0), 'time'] = 10.0
    return frame


def _noise(frame, background, rng, strength, category_rate=None, event_time_sd=None):
    from sugiyama_v2.releases import CONTINUOUS
    c = frame.copy(deep=True).reset_index(drop=True)
    scale = background[CONTINUOUS].std(ddof=0).to_numpy(float)
    c[CONTINUOUS] = c[CONTINUOUS].to_numpy(float) + rng.normal(size=(len(c),len(CONTINUOUS))) * scale * strength
    event = (c.onset.to_numpy(int) + c.death.to_numpy(int)) > 0
    sd = .10 + .5*strength if event_time_sd is None else event_time_sd
    c.loc[event, 'time'] = c.loc[event,'time'].to_numpy(float) + rng.normal(0, sd, int(event.sum()))
    rate = min(.3, .10*strength) if category_rate is None else category_rate
    for column in ('sex','smoking','prefecture'):
        change = rng.random(len(c)) < rate
        c.loc[change,column] = rng.choice(background[column].to_numpy(), int(change.sum()))
    return c


def make_candidate(target, background, specification, seed, schema):
    """Release generation depends on B/background only; never challenge or roles."""
    from sugiyama_v2 import releases
    rng = np.random.default_rng(seed)
    family, strength = specification['family'], float(specification['strength'])
    features = releases.FEATURES
    background = background[features].copy(deep=True).reset_index(drop=True)
    # Null family deliberately never validates/reads target.
    if family == 'independent':
        c = background.iloc[rng.integers(len(background),size=1049)].reset_index(drop=True)
        c = _noise(c, background, rng, strength)
    else:
        target = target[features].copy(deep=True).reset_index(drop=True)
        if family in ('resample','category_redraw'):
            c = target.iloc[rng.integers(len(target),size=1049)].reset_index(drop=True)
        elif family in ('copula','shifted'):
            c = releases._copula(target, rng)
        elif family == 'conditional':
            c = releases._conditional(target, background, rng)
        elif family == 'jitter':
            c = target
        else:
            raise ValueError(f'Unknown predeclared family: {family}')
        if family == 'shifted':
            columns = releases.CONTINUOUS
            signs = np.where(np.arange(len(columns)) % 2, -1., 1.)
            c[columns] = c[columns].to_numpy(float) + strength * signs * background[columns].std(ddof=0).to_numpy(float)
            c = _noise(c,background,rng,.08,event_time_sd=.3)
        elif family == 'category_redraw':
            c = _noise(c,background,rng,.08,category_rate=strength,event_time_sd=.3)
        else:
            c = _noise(c,background,rng,strength)
    return _censor(releases._finish(c,rng,schema))


def _synchronize_censoring(world):
    """One common change to synthetic birth process, applied before release fitting."""
    from sugiyama_v2.generate import FEATURES
    public, private = world['public'], world['private']
    pool = _censor(private['pool'])
    private['pool'] = pool
    ids = private['candidate_pool_ids']
    candidates = pool.iloc[ids][FEATURES].reset_index(drop=True)
    candidates.insert(0,'record_id',public['candidates'].record_id.to_numpy())
    public['candidates'] = candidates
    labels = candidates.copy()
    labels['is_rare'] = pool.iloc[ids].is_rare.to_numpy(int)
    public['self_labels'] = labels
    for team, truth in private['aia_truth'].items():
        truth = truth.copy()
        truth['time'] = pool.iloc[private['challenge_pool_ids'][team]].time.to_numpy(float)
        private['aia_truth'][team] = truth


def paired_demographic_slots(challenge, rng):
    """Project public cell counts minimally onto two symmetric 163-row sides."""
    q = challenge[['prefecture','sex','age']].copy().reset_index(drop=True)
    q['age5'] = ((q.age - 40) // 5).astype(int)
    groups = list(q.groupby(['prefecture','sex','age5'],sort=True))
    odd = [i for i,(_,group) in enumerate(groups) if len(group)%2]
    if len(q)!=326 or len(odd)%2:
        raise ValueError('Public challenge cannot be split into equal sides')
    round_up = set(rng.permutation(odd)[:len(odd)//2].tolist())
    slots, cells = [], []
    for i,(key,group) in enumerate(groups):
        n = len(group)//2 + int(i in round_up)
        ages = group.age.to_numpy(int).copy()
        if len(ages)<2*n:
            ages=np.r_[ages,rng.choice(ages)]
        elif len(ages)>2*n:
            ages=np.delete(ages,int(rng.integers(len(ages))))
        ages=ages[rng.permutation(len(ages))]
        for am,ac in zip(ages[:n],ages[n:]):
            slots.append({'prefecture':str(key[0]),'sex':int(key[1]),'age5':int(key[2]),'age_member':int(am),'age_control':int(ac)})
        cells.append({'prefecture':str(key[0]),'sex':int(key[1]),'age5':int(key[2]),'observed':len(group),'projected':2*n,'per_side':n,'difference':2*n-len(group)})
    if len(slots)!=163:
        raise AssertionError('Projected sides must have 163 rows')
    return slots, {'odd_cells':len(odd),'pooled_count_l1_change':sum(abs(x['difference']) for x in cells),'cells':cells}


def _assignment(original, desired_sex, desired_age, rng):
    from scipy.optimize import linear_sum_assignment
    cost = 25*(original.sex.to_numpy(int)[:,None] != np.asarray(desired_sex)[None,:])
    cost = cost + ((original.age.to_numpy(float)[:,None]-np.asarray(desired_age)[None,:])/5)**2
    cost += rng.uniform(0,1e-10,size=cost.shape)
    source,desired=linear_sum_assignment(cost)
    if not np.array_equal(source,np.arange(len(original))):
        raise AssertionError('Incomplete demographic assignment')
    return desired


def transport_public_demographics(world, actual_challenges, seed):
    """Transport only private synthetic persons; role labels never change.

    Synthetic role access locates paired people, but assignments use only public
    demographics. Both sides undergo the same transport. No conditional clinical
    donor for a missing rare/prefecture stratum is claimed to have been generated.
    """
    from sugiyama_v2.generate import FEATURES,QI
    public,hidden=world['public'],world['private']
    pool=hidden['pool'].copy(deep=True)
    before=pool.copy(deep=True)
    occurrence=Counter(int(pid) for ids in hidden['cohorts'].values() for pid in ids)
    shared=[pid for pid,count in occurrence.items() if count>1]
    summaries=[]
    for team,observed in sorted(actual_challenges.items()):
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(team),2026100914]))
        truth=hidden['aia_truth'][team]
        challenge_ids=np.asarray(hidden['challenge_pool_ids'][team],dtype=int)
        members=np.sort(challenge_ids[truth.label.to_numpy()=='member'])
        controls=np.sort(challenge_ids[truth.label.to_numpy()=='control'])
        # The inherited generator allocates members and their exact-key controls
        # in matching creation order; verify the keys rather than assuming it.
        for key in ('is_rare','onset','sex','prefecture'):
            if not np.array_equal(pool.loc[members,key].to_numpy(),pool.loc[controls,key].to_numpy()):
                raise AssertionError(f'Inherited member/control key differs: {team}/{key}')
        if not np.array_equal(((pool.loc[members,'age']-40)//5).to_numpy(),((pool.loc[controls,'age']-40)//5).to_numpy()):
            raise AssertionError('Inherited member/control age bins differ')
        slots,diagnostic=paired_demographic_slots(observed,rng)
        original_pairs=pool.loc[members].copy()
        original_pairs['age']=(pool.loc[members,'age'].to_numpy(float)+pool.loc[controls,'age'].to_numpy(float))/2
        assigned=_assignment(original_pairs, [s['sex'] for s in slots],
            [(s['age_member']+s['age_control'])/2 for s in slots],rng)
        for j,slot_index in enumerate(assigned):
            slot=slots[int(slot_index)]
            for pid,age_key in ((members[j],'age_member'),(controls[j],'age_control')):
                pool.loc[pid,'prefecture']=slot['prefecture']
                pool.loc[pid,'sex']=slot['sex']
                pool.loc[pid,'age']=slot[age_key]
        others=sorted(set(pid for pid in hidden['cohorts'][team] if occurrence[pid]==1)-set(members))
        if len(others)!=150:
            raise AssertionError('Expected 150 other private people')
        desired=observed[['prefecture','sex','age']].iloc[rng.integers(len(observed),size=150)].reset_index(drop=True)
        assigned_other=_assignment(pool.loc[others],desired.sex.to_numpy(),desired.age.to_numpy(),rng)
        for j,index in enumerate(assigned_other):
            row=desired.iloc[int(index)]
            pool.loc[others[j],'prefecture']=row.prefecture
            pool.loc[others[j],'sex']=int(row.sex)
            pool.loc[others[j],'age']=int(row.age)
        for key in ('prefecture','sex'):
            if not np.array_equal(pool.loc[members,key].to_numpy(),pool.loc[controls,key].to_numpy()):
                raise AssertionError('Transport must be symmetric')
        if not np.array_equal(((pool.loc[members,'age']-40)//5).to_numpy(),((pool.loc[controls,'age']-40)//5).to_numpy()):
            raise AssertionError('Transport must keep paired age bins equal')
        diagnostic.update({'target':int(team),'private_members_transported':163,'controls_transported':163,'other_private_transported':150})
        for side,ids in [('member',members),('control',controls),('other_private',others)]:
            diagnostic[side]={'prefecture_changed_fraction':float((before.loc[ids,'prefecture']!=pool.loc[ids,'prefecture']).mean()),
                'sex_changed_fraction':float((before.loc[ids,'sex']!=pool.loc[ids,'sex']).mean()),
                'age_mean_absolute_change':float(np.abs(before.loc[ids,'age']-pool.loc[ids,'age']).mean()),
                'age_max_absolute_change':int(np.abs(before.loc[ids,'age']-pool.loc[ids,'age']).max())}
        summaries.append(diagnostic)
    if not pool.loc[shared,FEATURES].equals(before.loc[shared,FEATURES]):
        raise AssertionError('Shared-person features changed during transport')
    if not pool[['pool_id','is_rare','onset','death','time']].equals(before[['pool_id','is_rare','onset','death','time']]):
        raise AssertionError('IDs or birth/outcome truth changed during transport')
    hidden['pool']=pool
    # Propagate changed public demographics. Truth outputs/onsets/rare labels stay
    # byte-for-byte equal because none of them were modified by this step.
    ids=hidden['candidate_pool_ids']
    candidates=pool.iloc[ids][FEATURES].reset_index(drop=True)
    candidates.insert(0,'record_id',public['candidates'].record_id.to_numpy())
    public['candidates']=candidates
    public['self_labels']=candidates.assign(is_rare=pool.iloc[ids].is_rare.to_numpy(int))
    for team in range(30):
        challenge=pool.iloc[hidden['challenge_pool_ids'][team]][QI].reset_index(drop=True)
        challenge.insert(0,'challenge_row_id',np.arange(326,dtype=int))
        if team==10:
            public['self_challenge']=challenge
        if team in public['targets']:
            public['targets'][team]['challenge']=challenge
    return {'targets':summaries,'shared_people_unchanged':True,'shared_person_count':len(shared),
        'ids_birth_labels_onset_death_time_unchanged':True,
        'method':'Symmetric public demographic transport; clinical conditional relations are an approximation.'}


def background_demographic_diagnostics(actual, synthetic, self_challenge):
    """Public demographic audit only; never used by release selection."""
    own=str(self_challenge.prefecture.iloc[0])
    result={'self_private_prefecture':own,'diagnostic_only':True,'distributions':{}}
    for name,frame in [('actual',actual),('synthetic',synthetic)]:
        pref=frame.prefecture.astype(str).value_counts(normalize=True)
        result[name]={'rows':len(frame),'max_prefecture_fraction':float(pref.max()),
            'self_prefecture_fraction':float(pref.get(own,0.0)),
            'age_mean':float(frame.age.mean()),'age_sd':float(frame.age.std(ddof=0))}
    for column in ('prefecture','sex','age'):
        observed=actual[column].astype(str).value_counts(normalize=True)
        generated=synthetic[column].astype(str).value_counts(normalize=True)
        keys=sorted(set(observed.index)|set(generated.index))
        result['distributions'][column]={'actual':{k:float(observed.get(k,0)) for k in keys},
            'synthetic':{k:float(generated.get(k,0)) for k in keys},
            'total_variation':float(sum(abs(observed.get(k,0)-generated.get(k,0)) for k in keys)/2)}
    return result


def generate_world(seed, split=None, *, root=None, diagnostic_callback=None):
    """Return an existing export_world-compatible public/private world dictionary."""
    from sugiyama_v2 import generate as gen
    from sugiyama_v2 import releases
    protocol = load_protocol()
    if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)):
        raise ValueError('Seed must be an integer reserved by protocol')
    required_split, role = _reserved_split(int(seed),protocol)
    if split is None:
        split = required_split
    if split != required_split:
        raise ValueError('Seed/split does not match the fixed protocol')
    root = ROOT if root is None else Path(root).resolve()
    verification = verify_original_inputs(root)
    actual, scales, public_hashes = public_targets(root,protocol)
    source = root / 'main_round/anonymization/team10/participant_data'
    donors = pd.read_csv(source/'B_self.csv',float_precision='round_trip')
    empirical = pd.read_csv(source/'A_bg.csv',float_precision='round_trip')
    schema = json.loads((source/'schema.json').read_text(encoding='utf-8-sig'))
    # Placeholder C is never exported or attacked; every target is replaced below.
    world = gen.build_world(seed,profile='mixed',split=split,donors=donors,
        background_source=empirical,schema=schema,
        release_factory=lambda target,*args: target[gen.FEATURES].copy())
    _synchronize_censoring(world)
    demographic_inputs={}
    for path in sorted((root/'main_round/attack/targets').glob('aia_challenge_*.csv')):
        team=int(path.stem.split('_')[-1])
        demographic_inputs[team]=pd.read_csv(path,float_precision='round_trip')
        public_hashes[path.relative_to(root).as_posix()]=_sha(path)
    transport=transport_public_demographics(world,demographic_inputs,int(seed))
    labeled, empirical_split, _, _ = gen._partition_donors(donors,empirical,split)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed),2026100903]))
    rare = rng.binomial(1,24/1049,5143)
    onset_rates = {r:float(labeled.loc[labeled.is_rare==r,'onset'].mean()) for r in (0,1)}
    onset = np.array([rng.binomial(1,onset_rates[int(r)]) for r in rare],dtype=int)
    background, _ = gen._kernel(labeled,np.column_stack([rare,onset]),empirical_split,rng,schema)
    background = _censor(background[gen.FEATURES])
    world['public']['background'] = background
    background_audit=background_demographic_diagnostics(empirical,background,demographic_inputs[10])
    fitted, selected_profiles, grid_records = [], [], []
    for team in gen.ACTIVE_TARGETS:
        target = world['private']['pool'].iloc[world['private']['cohorts'][team]][gen.FEATURES].reset_index(drop=True)
        challenge = world['public']['targets'][team]['challenge']
        best = None
        for index,specification in enumerate(protocol['grid']):
            candidate_seed = int(np.random.SeedSequence([int(seed),team,index,2026100904]).generate_state(1)[0])
            c = make_candidate(target,background,specification,candidate_seed,schema)
            p = profile_tables(c,challenge,background)
            loss,residuals = public_fit_loss(p,actual[team],scales,protocol)
            record = {'target':int(team),'grid_index':index,**specification,'seed':candidate_seed,
                      'public_profile_loss':loss if np.isfinite(loss) else None,'profile':p,'residuals':residuals}
            grid_records.append(record)
            if best is None or loss < best[0]:
                best=(loss,index,c,p,record)
        if best is None or not np.isfinite(best[0]):
            raise ValueError(f'No finite candidate for target {team}; retain this failure')
        loss,index,c,p,record=best
        world['public']['targets'][team]['release']=c
        world['metadata']['target_styles'][str(team)]='profile_v4_'+record['family']
        selected_profiles.append({'name':f'C_{team}','cohort':int(team),**p})
        fitted.append({'target':int(team),'grid_index':index,'family':record['family'],
                       'strength':record['strength'],'public_profile_loss':loss,'residuals':record['residuals']})
        if diagnostic_callback:
            diagnostic_callback({'target':int(team),'family':record['family'],'loss':loss})
    coverage=build_coverage_report(selected_profiles,list(actual.values()))
    world['metadata'].update({
        'generator':'public_profile_joint_v4','version':4,'profile':'public_main_profile_v4',
        'protocol_version':protocol['protocol_version'],'protocol_sha256':PROTOCOL_SHA256,
        'reserved_seed_role':role,'generator_source_sha256':_sha(Path(__file__)),
        'inherited_generator_sha256':_sha(gen.__file__),'inherited_releases_sha256':_sha(releases.__file__),
        'source_verification':verification,'actual_public_inputs_sha256':public_hashes,
        'background_generation':'Shared clinical labeled-donor kernel with independent draws; candidate/private demographic transport is applied after kernel generation; labels removed from background.',
        'control_generation':'Same exact-key clinical donor kernel, followed by symmetric private demographic transport using mean ages of both roles.',
        'background_demographics':background_audit,
        'censoring_generation':'All onset=death=0 rows have time=10 across pool/truth/public tables.',
        'profile_metric_scales':scales,'public_profile_fit':fitted,
        'demographic_transport':transport,
        'public_profile_family_counts':dict(Counter(x['family'] for x in fitted)),
        'public_profile_coverage':coverage,
        'selection_uses_official_feedback':False,'selection_uses_attack_predictions':False,
        'selection_uses_membership_or_aia_labels':False,
        'limitations':[x for x in world['metadata']['limitations'] if not x.startswith('Private prefecture concentration is not modeled')]+protocol['limitations'],
    })
    # Kept private; existing export intentionally writes only fixed public tables.
    world['private']['public_profile_diagnostics']={
        'actual_profiles':list(actual.values()),'selected_profiles':selected_profiles,
        'grid_profiles':grid_records,'coverage':coverage,'demographic_transport':transport,
        'background_demographics':background_audit,
    }
    return world


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--split',choices=['dev','holdout'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    output=args.output.resolve()
    base=args.root.resolve()/'main_round/mock_test/local-output'
    if output.exists() or not output.is_relative_to(base) or output==base:
        raise ValueError('Output must be a new child of mock_test/local-output')
    world=generate_world(args.seed,args.split,root=args.root,
        diagnostic_callback=lambda row:print(json.dumps(row),flush=True))
    from sugiyama_v2.io import export_world,save_json
    export_world(world,output)
    save_json(output/'private/public_profile_grid.json',world['private']['public_profile_diagnostics'])
    # The extra diagnostic file must participate in the immutable case manifest.
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    manifest['files_sha256']['private/public_profile_grid.json']=_sha(output/'private/public_profile_grid.json')
    save_json(output/'manifest.json',manifest)
    print(str(output),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
