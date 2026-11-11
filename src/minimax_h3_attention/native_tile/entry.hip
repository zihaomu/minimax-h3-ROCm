// Uses the unchanged private copy of the original native kernel definitions.
#include "native_prefix.cuh"

template<int BR>
void launch_tile(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                 torch::Tensor out, torch::Tensor qs, torch::Tensor ks,
                 torch::Tensor vs, int64_t valid_n) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 block(BR), grid(q.size(2)/BR, q.size(1), q.size(0));
  qk_int8_sv_f8_native_2q_kernel<64,128,0,8,true,BR,true,false,__hip_bfloat16>
      <<<grid,block,0,stream>>>(
      q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), v.data_ptr<uint8_t>(),
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
      qs.data_ptr<float>(), ks.data_ptr<float>(), vs.data_ptr<float>(),
      q.size(0),q.size(2),valid_n,q.size(1),k.size(1),
      q.stride(0),q.stride(2),q.stride(1),
      k.stride(0),k.stride(2),k.stride(1),
      v.stride(0),v.stride(2),v.stride(1),
      out.stride(0),out.stride(2),out.stride(1),
      qs.stride(0),qs.stride(1),ks.stride(0),ks.stride(1),
      1,static_cast<float>(1.0/std::sqrt(128.0)),false);
  hip_kernel_launch_check();
}

void tile(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
          torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, int64_t valid_n, int br) {
  TORCH_CHECK(br==128 || br==256 || br==512,"Unsupported BR");
  TORCH_CHECK(q.dim()==4 && q.size(0)==1 && q.size(1)==7 && q.size(3)==128,"Expected H3 7-head HND");
  TORCH_CHECK(q.size(2)%br==0 && k.size(2)%64==0 && valid_n>0 && valid_n<=k.size(2),"Padding / valid N mismatch");
  TORCH_CHECK(q.scalar_type()==torch::kInt8 && k.scalar_type()==torch::kInt8 && v.scalar_type()==torch::kUInt8,"Packed dtype mismatch");
  TORCH_CHECK(out.scalar_type()==torch::kBFloat16 && out.sizes()==q.sizes(),"BF16 output mismatch");
  TORCH_CHECK(v.size(2)==128 && v.size(3)==k.size(2),"Transposed V mismatch");
  for(const auto& x : {q,k,v,out,qs,ks,vs}) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.device()==q.device(),"Packed tensors must be contiguous on the same GPU");
  }
  if(br==128) launch_tile<128>(q,k,v,out,qs,ks,vs,valid_n);
  else if(br==256) launch_tile<256>(q,k,v,out,qs,ks,vs,valid_n);
  else launch_tile<512>(q,k,v,out,qs,ks,vs,valid_n);
}

template<int BR>
pybind11::dict resource() {
  hipFuncAttributes attr{};
  const auto fn=qk_int8_sv_f8_native_2q_kernel<64,128,0,8,true,BR,true,false,__hip_bfloat16>;
  TORCH_CHECK(hipFuncGetAttributes(&attr,reinterpret_cast<const void*>(fn))==hipSuccess,"Cannot query attributes");
  int blocks=0;
  TORCH_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,BR,0)==hipSuccess,"Cannot query occupancy");
  pybind11::dict r;
  r["BR"]=BR;r["threads_per_block"]=BR;r["numRegs"]=attr.numRegs;
  r["shared_bytes"]=attr.sharedSizeBytes;r["local_bytes"]=attr.localSizeBytes;
  r["max_threads_per_block"]=attr.maxThreadsPerBlock;r["max_active_blocks_per_multiprocessor"]=blocks;
  return r;
}

pybind11::dict resources(int br) {
  if(br==128) return resource<128>();
  if(br==256) return resource<256>();
  TORCH_CHECK(br==512,"Unsupported BR");return resource<512>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("tile",&tile);m.def("resources",&resources);}
