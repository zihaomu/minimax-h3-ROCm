// Instantiate the unchanged Ix kernel with a different query-to-wave mapping.
#include "native_prefix.cuh"

template<int BR, int QG>
using KernelTag = std::integral_constant<int, BR * 10 + QG>;

template<int BR, int QG>
void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
            torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, int64_t n) {
  const dim3 block(BR * (2 / QG)), grid(q.size(2) / BR, q.size(1), q.size(0));
  qk_int8_sv_f8_native_2q_kernel<64,128,0,8,true,BR,true,false,__hip_bfloat16,
      int8_t,false,int8_t,uint8_t,false,false,0,false,false,QG>
      <<<grid,block,0,at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<int8_t>(),k.data_ptr<int8_t>(),v.data_ptr<uint8_t>(),
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
      qs.data_ptr<float>(),ks.data_ptr<float>(),vs.data_ptr<float>(),
      q.size(0),q.size(2),n,q.size(1),k.size(1),
      q.stride(0),q.stride(2),q.stride(1),k.stride(0),k.stride(2),k.stride(1),
      v.stride(0),v.stride(2),v.stride(1),out.stride(0),out.stride(2),out.stride(1),
      qs.stride(0),qs.stride(1),ks.stride(0),ks.stride(1),1,
      static_cast<float>(1.0/std::sqrt(128.0)),false);
  hip_kernel_launch_check();
}

template<int BR, int QG>
pybind11::dict resource() {
  const auto fn=qk_int8_sv_f8_native_2q_kernel<64,128,0,8,true,BR,true,false,
      __hip_bfloat16,int8_t,false,int8_t,uint8_t,false,false,0,false,false,QG>;
  hipFuncAttributes a{};
  TORCH_CHECK(hipFuncGetAttributes(&a,reinterpret_cast<const void*>(fn))==hipSuccess);
  int blocks=0;
  TORCH_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,BR*(2/QG),0)==hipSuccess);
  pybind11::dict r;
  r["BR"]=BR;r["query_groups_per_wave"]=QG;r["threads_per_block"]=BR*(2/QG);
  r["numRegs"]=a.numRegs;r["shared_bytes"]=a.sharedSizeBytes;r["local_bytes"]=a.localSizeBytes;
  r["theoretical_active_blocks_per_multiprocessor"]=blocks;
  return r;
}

void tile(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
          torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, int64_t n, int variant) {
  TORCH_CHECK(q.dim()==4 && q.size(0)==1 && q.size(1)==7 && q.size(3)==128);
  TORCH_CHECK(q.scalar_type()==torch::kInt8 && k.scalar_type()==torch::kInt8 && v.scalar_type()==torch::kUInt8);
  TORCH_CHECK(out.scalar_type()==torch::kBFloat16 && out.sizes()==q.sizes());
  TORCH_CHECK(q.size(2)%512==0 && k.size(2)%64==0 && n>0 && n<=k.size(2));
  TORCH_CHECK(v.size(2)==128 && v.size(3)==k.size(2));
  for(const auto& x:{q,k,v,out,qs,ks,vs})
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.device()==q.device());
  if(variant==0) launch<128,2>(q,k,v,out,qs,ks,vs,n);
  else if(variant==1) launch<512,2>(q,k,v,out,qs,ks,vs,n);
  else if(variant==2) launch<128,1>(q,k,v,out,qs,ks,vs,n);
  else if(variant==3) launch<256,1>(q,k,v,out,qs,ks,vs,n);
  else TORCH_CHECK(false,"Unknown isolated spill probe variant");
}

pybind11::dict resources(int variant) {
  if(variant==0) return resource<128,2>();
  if(variant==1) return resource<512,2>();
  if(variant==2) return resource<128,1>();
  TORCH_CHECK(variant==3);
  return resource<256,1>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("tile",&tile);m.def("resources",&resources);}
