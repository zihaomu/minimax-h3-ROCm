# Ported from the locally qualified S11 adapter; parameter/scale semantics retained.
"""Opt-in immutable CPU parameter backing for a single inference-only patcher.

Keep original CPU storages before loading. On a root .to(cpu), finish device
work, restore those parameters, then run the original .to for buffers and all
normal ModelPatcher cleanup. No tensor/global class or engine file is patched.
"""
import dataclasses
import functools
import time

import torch
from comfy_kitchen.tensor import QuantizedTensor


def parameter_parts(parameter):
    if isinstance(parameter, QuantizedTensor):
        tensors={'qdata':parameter._qdata};meta={'layout':parameter._layout_cls,
            'params_type':type(parameter._params).__qualname__}
        for field in dataclasses.fields(parameter._params):
            value=getattr(parameter._params,field.name)
            if isinstance(value,torch.Tensor):tensors[field.name]=value
            else:meta[field.name]=str(value) if isinstance(value,torch.dtype) else value
    else:
        assert type(parameter) in (torch.Tensor,torch.nn.Parameter),type(parameter)
        tensors={'data':parameter};meta={}
    return tensors,meta


def signature(parameter):
    tensors,meta=parameter_parts(parameter)
    return {'shape':tuple(parameter.shape),'dtype':str(parameter.dtype),
        'requires_grad':parameter.requires_grad,'metadata':meta,
        'parts':{key:(tuple(t.shape),tuple(t.stride()),str(t.dtype)) for key,t in tensors.items()}}


def version(tensor):
    try:return tensor._version
    except RuntimeError:return None  # Inference tensors have no version counter.


def cpu_parameter_reference(parameter):
    tensors,_=parameter_parts(parameter)
    assert all(t.device.type=='cpu' for t in tensors.values())
    if isinstance(parameter,QuantizedTensor):
        # Avoid dequantization and .detach()'s implicit metadata cloning.
        data=QuantizedTensor(parameter._qdata.detach(),parameter._layout_cls,
                             dataclasses.replace(parameter._params))
    else:data=parameter.detach()
    # Keep a raw tensor, never the Parameter object later installed in Module.
    # Module._apply can change Parameter.data in place on the next H2D load.
    return data


class CPUWeightReuse:
    def __init__(self,patcher):
        assert torch.is_inference_mode_enabled(),'Immutable backing supports inference_mode only'
        self.patcher=patcher;self.root=patcher.model;self.rows=[];self.enabled=False;self.closed=False
        self.uuid=patcher.patches_uuid;self.slots=[];refs={}
        self._guard()
        self.original_to=self.root.to;self.had_local='to' in vars(self.root);self.local_to=vars(self.root).get('to')
        for module_name,module in self.root.named_modules():
            for name,param in module._parameters.items():
                if param is None:continue
                if id(param) not in refs:refs[id(param)]=cpu_parameter_reference(param)
                ref=refs[id(param)];parts,_=parameter_parts(ref)
                self.slots.append((module_name,module,name,ref,signature(param),
                    {key:version(t) for key,t in parts.items()}))
        assert self.slots
        self.topology=[(id(m),key) for _,m,key,_,_,_ in self.slots]
        unique={}
        for _,_,_,ref,_,_ in self.slots:
            for t in parameter_parts(ref)[0].values():
                unique[(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes())]=t.untyped_storage().nbytes()
        self.retained_CPU_storage_bytes=sum(unique.values())
        @functools.wraps(self.original_to)
        def to(*args,**kwargs):
            device,dtype,_,_=torch._C._nn._parse_to(*args,**kwargs)
            if not self.enabled or device is None or device.type!='cpu':return self.original_to(*args,**kwargs)
            assert dtype is None,'CPU reuse cannot change weight dtype'
            self._guard();self._validate_slots()
            devices=sorted({t.device for _,m,key,_,_,_ in self.slots
                for t in parameter_parts(m._parameters[key])[0].values() if t.device.type!='cpu'},key=str)
            assert all(d.type=='cuda' for d in devices),devices
            if not devices:return self.original_to(*args,**kwargs)
            start=time.perf_counter_ns()
            # Replacing references removes the D2H copy's implicit completion.
            # This one barrier per offload is necessary and stays inside timing.
            for d in devices:torch.cuda.synchronize(d)
            ready=time.perf_counter_ns();avoided=self._restore_parameters()
            restored=time.perf_counter_ns();result=self.original_to(*args,**kwargs)
            self.rows.append({'index':len(self.rows),'parameters':len(self.slots),
                'device_parameter_payload_bytes_replaced':avoided,'D2H_parameter_copy_performed':False,
                'completion_barriers':len(devices),'completion_barrier_seconds':(ready-start)/1e9,
                'reference_restore_seconds':(restored-ready)/1e9,
                'root_to_total_seconds':(time.perf_counter_ns()-start)/1e9,
                'buffers_moved_by_original_to':True})
            return result
        self.root.to=to

    def _guard(self):
        assert not self.closed and torch.is_inference_mode_enabled()
        assert self.patcher.patches_uuid==self.uuid
        for name in ('patches','backup','backup_buffers','hook_backup','forced_hooks','weight_wrapper_patches'):
            assert not getattr(self.patcher,name,None),('Unsupported mutable patch state',name)

    def _validate_slots(self):
        topology=[(id(m),key) for m in self.root.modules() for key,p in m._parameters.items() if p is not None]
        assert topology==self.topology,'Parameter registration changed'
        for module_name,module,key,ref,expected,versions in self.slots:
            assert signature(module._parameters[key])==expected,(module_name,key,'Parameter representation changed')
            parts,_=parameter_parts(ref)
            assert all(t.device.type=='cpu' for t in parts.values())
            assert {k:version(t) for k,t in parts.items()}==versions,'CPU backing was modified'

    def _restore_parameters(self):
        self._validate_slots();total=0;wrappers={}
        for _,module,key,ref,expected,_ in self.slots:
            total+=sum(t.numel()*t.element_size() for t in parameter_parts(module._parameters[key])[0].values()
                       if t.device.type!='cpu')
            if id(ref) not in wrappers:
                wrappers[id(ref)]=torch.nn.Parameter(ref,requires_grad=expected['requires_grad'])
            module._parameters[key]=wrappers[id(ref)]
        return total

    def set_enabled(self,value):
        self._guard();self.enabled=bool(value)

    def close(self):
        if self.closed:return
        if self.had_local:self.root.to=self.local_to
        else:delattr(self.root,'to')
        self.slots.clear();self.closed=True

    def record(self):
        return {'scope':'Immutable CPU parameter backing; buffers use original path; TE-only candidate',
            'parameters':len(self.slots),'retained_CPU_storage_bytes':self.retained_CPU_storage_bytes,
            'version_counter_limitation':'Inference tensors lack counters; qualification checks full bytes and fixed source, not arbitrary mutation safety',
            'offloads':self.rows,'enabled':self.enabled}
