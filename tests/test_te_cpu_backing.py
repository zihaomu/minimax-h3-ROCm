"""CPU checks for immutable backing and rejection of unsupported mutations."""
from types import SimpleNamespace
import unittest

import torch
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout
from minimax_h3_amd_sp.te_cpu_backing import CPUWeightReuse,parameter_parts,signature


def fixture(model):
    return SimpleNamespace(model=model,patches_uuid='fixed',patches={},backup={},backup_buffers={},
        hook_backup={},forced_hooks=None,weight_wrapper_patches={})


class Reuse(unittest.TestCase):
    def test_plain_reference_restoration_keeps_original_bits_and_buffer_path(self):
        with torch.inference_mode():
            m=torch.nn.Linear(4,2);m.register_buffer('buffer',torch.ones(3))
            expected=m.weight.detach();pointer=expected.data_ptr();patcher=fixture(m)
            reuse=CPUWeightReuse(patcher)
            m.to('meta');self.assertEqual(m.weight.device.type,'meta')
            reuse._restore_parameters()
            self.assertEqual(m.weight.data_ptr(),pointer)
            self.assertTrue(torch.equal(m.weight,expected))
            self.assertEqual(m.buffer.device.type,'meta')
            reuse.close();self.assertNotIn('to',vars(m))

    def test_quantized_packed_and_all_scale_metadata_are_preserved(self):
        with torch.inference_mode():
            params=TensorCoreNVFP4Layout.Params(scale=torch.tensor(0.25),orig_dtype=torch.bfloat16,
                orig_shape=(16,16),block_scale=torch.arange(16,dtype=torch.uint8).reshape(16,1))
            q=QuantizedTensor(torch.arange(128,dtype=torch.uint8).reshape(16,8),
                'TensorCoreNVFP4Layout',params)
            m=torch.nn.Module();m.register_parameter('weight',torch.nn.Parameter(q,requires_grad=False))
            expected=signature(m.weight);raw={k:t.clone() for k,t in parameter_parts(m.weight)[0].items()}
            reuse=CPUWeightReuse(fixture(m))
            m._parameters['weight']=torch.nn.Parameter(m.weight.to('meta'),requires_grad=False)
            reuse._restore_parameters();self.assertEqual(signature(m.weight),expected)
            for key,t in parameter_parts(m.weight)[0].items():self.assertTrue(torch.equal(t,raw[key]))
            reuse.close()

    def test_CPU_backing_survives_repeated_parameter_data_replacement(self):
        with torch.inference_mode():
            m=torch.nn.Linear(4,2);expected=m.weight.detach().clone();r=CPUWeightReuse(fixture(m))
            for _ in range(4):
                m.to('meta');r._restore_parameters()
                self.assertTrue(torch.equal(m.weight,expected))
                # Simulate a loader changing Parameter.data in place, followed
                # by a layer replacing its registered Parameter on offload.
                m.weight.data=torch.zeros_like(m.weight)
                m._parameters['weight']=torch.nn.Parameter(expected.clone())
                r._validate_slots()
                r._restore_parameters();self.assertTrue(torch.equal(m.weight,expected))
            r.close()

    def test_patch_change_rejected_before_restore(self):
        with torch.inference_mode():
            patcher=fixture(torch.nn.Linear(2,2));r=CPUWeightReuse(patcher)
            patcher.patches['weight']=[1]
            with self.assertRaises(AssertionError):r.set_enabled(True)
            patcher.patches.clear();patcher.patches_uuid='different'
            with self.assertRaises(AssertionError):r.set_enabled(True)
            r.close()

    def test_topology_and_dtype_change_rejected(self):
        with torch.inference_mode():
            m=torch.nn.Linear(2,2);r=CPUWeightReuse(fixture(m));m.to(dtype=torch.float16)
            with self.assertRaises(AssertionError):r._restore_parameters()
            r.close()
            m=torch.nn.Linear(2,2);r=CPUWeightReuse(fixture(m));m.register_parameter('extra',torch.nn.Parameter(torch.ones(1)))
            with self.assertRaises(AssertionError):r._restore_parameters()
            r.close()

    def test_original_local_to_method_restored_after_error(self):
        with torch.inference_mode():
            m=torch.nn.Linear(2,2);original=m.to;m.to=original;r=CPUWeightReuse(fixture(m));r.set_enabled(True)
            with self.assertRaises(AssertionError):m.to(device='cpu',dtype=torch.float16)
            r.close();self.assertIs(m.to,original)

    def test_training_context_is_rejected(self):
        with self.assertRaises(AssertionError):CPUWeightReuse(fixture(torch.nn.Linear(2,2)))


if __name__=='__main__':unittest.main()
