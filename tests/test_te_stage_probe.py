"""CPU call-tree accounting and restoration of instance/class methods on errors."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec=importlib.util.spec_from_file_location('h3_te_stage_probe_test',Path(__file__).resolve().parents[1]/'tools/h3_te_stage_probe.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Root:
    def forward(self):return 7
class Model:
    def __init__(self,root):self.root=root
    def named_modules(self):return iter([('qwen.transformer',self.root),('qwen.transformer.layers.49',object())])
    def encode_token_weights(self,*_):return self.root.forward()
class Clip:
    def __init__(self,mm):self.patcher=object();self.cond_stage_model=Model(Root());self.mm=mm
    def tokenize(self,*_):return [1]
    def load_model(self,*_):return self.mm.load_models_gpu([])
    def encode_from_tokens(self,*_):self.load_model();return self.cond_stage_model.encode_token_weights([])
    def encode_from_tokens_scheduled(self,*_):return self.encode_from_tokens([])

def setup():
    mm=SimpleNamespace(unload_model_and_clones=lambda *_:None,load_models_gpu=lambda *_:None)
    clip=Clip(mm);video=SimpleNamespace(patcher=object());audio=SimpleNamespace(patcher=object());dit=object()
    probe=m.TEStageProbe(clip,video,audio,dit,Path('/unused'))
    return probe,clip,mm,video,audio,dit

class Stages(unittest.TestCase):
    def test_nested_stages_close_without_double_counting(self):
        probe,clip,mm,video,audio,dit=setup();probe.begin_request('test')
        with probe.observe(mm):
            mm.unload_model_and_clones(video.patcher);mm.unload_model_and_clones(audio.patcher)
            mm.unload_model_and_clones(dit);clip.tokenize('prompt');self.assertEqual(clip.encode_from_tokens_scheduled([]),7)
            mm.unload_model_and_clones(clip.patcher)
        for row in probe.rows:
            children=[r for r in probe.rows if r['parent']==row['id']]
            row['exclusive_wall_seconds']=row['inclusive_wall_seconds']-sum(r['inclusive_wall_seconds'] for r in children)
            self.assertGreaterEqual(row['exclusive_wall_seconds'],0.)
        roots=[r for r in probe.rows if r['parent'] is None]
        self.assertAlmostEqual(sum(r['exclusive_wall_seconds'] for r in probe.rows),sum(r['inclusive_wall_seconds'] for r in roots))
        self.assertEqual([r['label'] for r in probe.rows].count('forward'),1)
        self.assertFalse(probe.active);self.assertEqual(probe.stack,[])
        self.assertNotIn('forward',vars(clip.cond_stage_model.root));self.assertNotIn('load_model',vars(clip))
    def test_exception_restores_methods_and_keeps_failure(self):
        probe,clip,mm,*_=setup();original=mm.load_models_gpu;probe.begin_request('failure')
        def fail(*_):raise RuntimeError('original failure')
        clip.load_model=fail
        with self.assertRaisesRegex(RuntimeError,'original failure'):
            with probe.observe(mm):clip.encode_from_tokens_scheduled([])
        self.assertIs(clip.load_model,fail);self.assertIs(mm.load_models_gpu,original)
        self.assertFalse(probe.active);self.assertEqual(probe.stack,[])
        self.assertTrue(any(r.get('exception_type')=='RuntimeError' for r in probe.rows))
    def test_other_threads_do_not_enter_owner_tree(self):
        import threading
        probe,clip,mm,*_=setup();probe.begin_request('thread')
        with probe.observe(mm):
            t=threading.Thread(target=clip.load_model);t.start();t.join()
        self.assertEqual(probe.rows,[])
    def test_record_rejects_missing_forward_or_broken_coverage(self):
        record={'diagnostic_only':True,'timing_eligible':False,'cuda_synchronizations_added':0,
            'TE_wall_seconds_original_scope':1.,'spans':[],'covered_root_seconds':0.,
            'unwrapped_and_instrumentation_seconds':1.,'TE_start_ns':0,'TE_stop_ns':10}
        with self.assertRaises(AssertionError):m.validate_record(record,1.)
    def test_event_pair_is_reused_and_read_after_request(self):
        import tempfile,time
        class Event:
            created=0
            def __init__(self):Event.created+=1
            def record(self):pass
            def query(self):return True
            def elapsed_time(self,other):return 2.5
        probe,clip,mm,video,audio,dit=setup();probe.events=Event
        with tempfile.TemporaryDirectory() as tmp:
            probe.base_dest=Path(tmp);probe.prepare_event_pool()
            for role in ('one','two'):
                probe.begin_request(role);start=time.perf_counter_ns()
                with probe.observe(mm):
                    mm.unload_model_and_clones(video.patcher);mm.unload_model_and_clones(audio.patcher)
                    mm.unload_model_and_clones(dit);clip.tokenize('prompt');clip.encode_from_tokens_scheduled([])
                    mm.unload_model_and_clones(clip.patcher)
                stop=time.perf_counter_ns();probe.stage_finished(start,stop,(stop-start)/1e9)
                self.assertTrue(m.validate_record(probe.record,(stop-start)/1e9))
                self.assertTrue(probe.record['event_pair_preallocated'])
            self.assertEqual(Event.created,2)

if __name__=='__main__':unittest.main()
