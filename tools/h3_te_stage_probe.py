# Consumer-owned CPU stage diagnostics, ported from the original S10 probe.
"""Scoped host-wall call tree for the unchanged Standard TE block.

No device synchronization is added. Deferred CUDA event spans are not pure
kernel time: they can include host submission gaps and exclude other streams.
"""
from contextlib import contextmanager
import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time


def host_snapshot():
    start=time.perf_counter_ns();nodes={};errors=[]
    try:
        for line in Path('/proc/self/numa_maps').read_text().splitlines():
            for field in line.split():
                if field.startswith('N') and '=' in field and field[1:field.index('=')].isdigit():
                    key,count=field.split('=');nodes[key]=nodes.get(key,0)+int(count)
    except OSError as e:errors.append(str(e))
    return {'pid':os.getpid(),'native_thread_id':threading.get_native_id(),
        'current_CPU':ctypes.CDLL(None).sched_getcpu(),
        'CPU_affinity':sorted(os.sched_getaffinity(0)),'resident_pages_by_NUMA_node':nodes,
        'page_size':os.sysconf('SC_PAGE_SIZE'),'loadavg':list(os.getloadavg()),
        'scope':'whole process mappings; not per-model page placement',
        'errors':errors,'collection_seconds':(time.perf_counter_ns()-start)/1e9}


def cpu_counters():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return {'process_CPU_ns':time.process_time_ns(),'thread_CPU_ns':time.thread_time_ns(),
        'minor_faults':usage.ru_minflt,'major_faults':usage.ru_majflt,
        'voluntary_context_switches':usage.ru_nvcsw,'involuntary_context_switches':usage.ru_nivcsw}


class TEStageProbe:
    def __init__(self,clip,video_vae,audio_vae,model,destination,event_factory=None):
        self.clip=clip;self.mm=None;self.base_dest=Path(destination);self.dest=self.base_dest
        self.labels={id(video_vae.patcher):'unload_video_vae',id(audio_vae.patcher):'unload_audio_vae',
                     id(model):'unload_dit',id(clip.patcher):'offload_text_encoder'}
        assert len(self.labels)==4
        roots=[(n.rsplit('.layers.49',1)[0],m) for n,m in clip.cond_stage_model.named_modules() if n.endswith('.layers.49')]
        assert len(roots)==1
        self.forward_name=roots[0][0]
        self.forward_module=dict(clip.cond_stage_model.named_modules())[self.forward_name]
        self.events=event_factory;self.pool=None;self.active=False;self.stack=[];self.rows=[];self.event_pairs=[]

    def prepare_event_pool(self):
        assert not self.active
        if self.events and self.pool is None:
            self.pool=(self.events(),self.events())
            self.pool[0].record();self.pool[1].record()
        return self.pool

    def begin_request(self,role):
        assert not self.active
        self.role=role;self.rows=[];self.stack=[];self.event_pairs=[]
        self.dest=self.base_dest/role
        self.before=host_snapshot();self.after=None;self.record=None

    @contextmanager
    def span(self,label,gpu_events=False,metadata=None):
        parent=self.stack[-1] if self.stack else None;index=len(self.rows)
        row={'id':index,'parent':parent,'label':label,'metadata':metadata or {},
             'clock_domain':'host_perf_counter_ns','time_basis':'CPU host-call interval'}
        self.rows.append(row);self.stack.append(index)
        counters=cpu_counters();row['start_ns']=time.perf_counter_ns();pair=None
        try:
            if gpu_events and self.events:
                pair=self.pool
                assert pair is not None and pair[0].query() and pair[1].query(), 'Event pool not ready'
                pair[0].record()
            yield row
        except BaseException as e:
            row['exception_type']=type(e).__name__;raise
        finally:
            if pair:
                pair[1].record();self.event_pairs.append((index,pair))
                row['forward_end_event_ready_at_host_return']=bool(pair[1].query())
            row['stop_ns']=time.perf_counter_ns();after=cpu_counters()
            row['inclusive_wall_seconds']=(row['stop_ns']-row['start_ns'])/1e9
            row['CPU_counter_delta']={key:after[key]-value for key,value in counters.items()}
            assert self.stack.pop()==index

    def _wrapper(self,original,label,gpu_events=False,metadata=None):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            if not self.active or threading.get_ident()!=self.owner_thread:return original(*args,**kwargs)
            name=label(*args,**kwargs) if callable(label) else label
            with self.span(name,gpu_events,metadata) as row:
                pass  # No tensor payload or metadata walk in the timed path.
                value=original(*args,**kwargs)
                if name=='encode_scheduled' and self.event_pairs:
                    row['forward_end_event_ready_after_encode']=bool(self.event_pairs[-1][1][1].query())
                return value
        return wrapped

    @contextmanager
    def observe(self,mm):
        assert not self.active;self.mm=mm;self.owner_thread=threading.get_ident();patched=[]
        def patch(obj,key,label,gpu_events=False,metadata=None):
            original=getattr(obj,key);local=key in vars(obj);value=vars(obj).get(key)
            setattr(obj,key,self._wrapper(original,label,gpu_events,metadata));patched.append((obj,key,local,value))
        try:
            patch(mm,'unload_model_and_clones',lambda model,*a,**k:self.labels.get(id(model),'unload_other_model'))
            patch(mm,'load_models_gpu','load_models_gpu')
            patch(self.clip,'tokenize','tokenize')
            patch(self.clip,'load_model','load_text_encoder')
            patch(self.clip,'encode_from_tokens_scheduled','encode_scheduled')
            patch(self.clip,'encode_from_tokens','encode_from_tokens')
            patch(self.clip.cond_stage_model,'encode_token_weights','encode_tokens')
            patch(self.forward_module,'forward','forward',True,{'module':self.forward_name})
            self.active=True;yield
        finally:
            self.active=False
            for obj,key,local,value in reversed(patched):
                if local:setattr(obj,key,value)
                else:delattr(obj,key)

    def stage_finished(self,start_ns,stop_ns,wall_seconds,request_completed=True):
        assert not self.active and not self.stack
        for row in self.rows:
            children=[r for r in self.rows if r['parent']==row['id']]
            row['exclusive_wall_seconds']=row['inclusive_wall_seconds']-sum(r['inclusive_wall_seconds'] for r in children)
            assert row['exclusive_wall_seconds']>=-1e-9
        roots=[r for r in self.rows if r['parent'] is None]
        covered=sum(r['inclusive_wall_seconds'] for r in roots)
        self.record={'schema_version':1,'role':self.role,'diagnostic_only':True,'timing_eligible':False,
            'TE_start_ns':start_ns,'TE_stop_ns':stop_ns,'TE_wall_seconds_original_scope':wall_seconds,
            'covered_root_seconds':covered,'unwrapped_and_instrumentation_seconds':wall_seconds-covered,
            'spans':self.rows,'host_before':self.before,'cuda_synchronizations_added':0,
            'event_pair_preallocated':self.pool is not None,
            'event_pool_initialization_outside_requests':True,
            'cuda_events_enabled':bool(self.events),
            'forward_cuda_event_semantics':'current-stream elapsed span; includes host gaps, not pure compute, not all-stream transfer time',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        assert self.record['unwrapped_and_instrumentation_seconds']>=-1e-5
        self.after=host_snapshot();self.record['host_after_request']=self.after
        self.record['post_snapshot_scope']='whole request end; not immediately after TE'
        return self.persist(request_completed)

    def persist(self,request_completed):
        assert self.record is not None
        for index,(begin,end) in self.event_pairs:
            ready=begin.query() and end.query()
            self.rows[index]['cuda_event_ready']=ready
            self.rows[index]['current_stream_elapsed_ms']=begin.elapsed_time(end) if ready else None
        self.record['request_completed']=request_completed
        self.dest.mkdir(parents=True,exist_ok=True)
        path=self.dest/'te-breakdown.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.record,indent=2,allow_nan=False)+'\n');tmp.replace(path)
        return {'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'diagnostic_only':True,'timing_eligible':False}


def validate_record(record,te_seconds):
    assert record['diagnostic_only'] and not record['timing_eligible'] and record['cuda_synchronizations_added']==0
    assert record['TE_wall_seconds_original_scope']==te_seconds
    rows=record['spans'];roots=[r for r in rows if r['parent'] is None]
    assert abs(sum(r['exclusive_wall_seconds'] for r in rows)-sum(r['inclusive_wall_seconds'] for r in roots))<1e-7
    assert abs(record['covered_root_seconds']+record['unwrapped_and_instrumentation_seconds']-te_seconds)<1e-7
    for row in rows:
        assert record['TE_start_ns']<=row['start_ns']<=row['stop_ns']<=record['TE_stop_ns']
        assert 'exception_type' not in row
        if row['parent'] is not None:
            p=rows[row['parent']];assert p['start_ns']<=row['start_ns']<=row['stop_ns']<=p['stop_ns']
    labels=[r['label'] for r in rows]
    for name in ('unload_video_vae','unload_audio_vae','unload_dit','tokenize','load_text_encoder',
                 'encode_scheduled','encode_from_tokens','encode_tokens','forward','offload_text_encoder'):
        assert labels.count(name)==1,(name,labels)
    if record.get('request_completed') and record['cuda_events_enabled']:
        forward=next(r for r in rows if r['label']=='forward')
        assert forward['cuda_event_ready'] and forward['current_stream_elapsed_ms']>=0
    return True


def tensor_summary(value):
    rows=[]
    def walk(x):
        if hasattr(x,'shape') and hasattr(x,'dtype') and hasattr(x,'device'):
            rows.append({'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device)})
        elif isinstance(x,(tuple,list)):
            for item in x:walk(item)
        elif isinstance(x,dict):
            for item in x.values():walk(item)
    walk(value);return rows[:12]
