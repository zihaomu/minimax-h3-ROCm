# MiniMax H3 ROCm

基于 [Kaihui-AMD/ComfyUI-MiniMaxH3-MultiGPU](https://github.com/Kaihui-AMD/ComfyUI-MiniMaxH3-MultiGPU) 的 AMD R9700 双卡推理项目。
使用 ComfyUI 的 H3模型、token / QKV head分片和RCCL通信，提供八种可明确选择的attention。
目标仓库：[zihaomu/minimax-h3-ROCm](https://github.com/zihaomu/minimax-h3-ROCm)。

## 从哪里开始

| 目的 | 入口 |
|---|---|
| 修改推理核心 | `src/minimax_h3_amd_sp/` |
| 修改或选择attention | `src/minimax_h3_attention/` |
| 修改ComfyUI节点 | `integrations/comfyui/nodes.py` |
| 查看配置、模型清单、工作流 | `configs/` |
| 查看关键结果 | [公开结果摘要](docs/RESULTS.md) |
| 继续本地研究 | `vibe/README.md`（本地保留，不上传） |

```text
minimax_h3/
├── src/                       # 推理和attention源代码
├── integrations/comfyui/       # ComfyUI适配；根nodes.py仅转发
├── configs/                   # 环境锁、模型清单、workflows、provenance
├── docs/                      # GitHub关键结果，精简公开摘要
├── vibe/                      # 完整本地研究入口，不进入Git
├── third_party/               # 原始第三方源码、许可证和来源说明
├── tools/                     # CPU检查、原生构建与基准适配
├── tests/                     # 配置、分片数学与路径合同
├── models/                    # 本地权重，不进入Git
├── .runtime/                  # 本机引擎、依赖、缓存与分析环境，不进入Git
├── runs/                      # 后续输出，不进入Git
└── archive/                   # 旧代码、实验与原始证据，不进入Git
```

## 本地检查

以下命令不占GPU，不加载模型正文：

```bash
python3 tools/check_project.py
python3 tools/check_models.py
python3 tools/show_config.py
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_attention_config.py'
```

Python源码统一通过 `src/` 导入，两个rank和构建器使用相同路径约定。
[工作流示例](configs/workflows/two_gpu_api.json)是生成结构示例，不含正式benchmark所需的完整动态预览验收。

## 模型与环境

权重保留在 `models/Comfy-Org-MiniMax-H3/` 的diffusion_models、text_encoders、vae目录。
[模型清单](configs/model-manifest.json)记录revision、大小和已有SHA；没有随代码上传权重。
[ComfyUI模型路径](configs/extra_model_paths.yaml)指向同一份共享模型。

精确ROCm10.1 / Torch / Triton / image锁见 [runtime-lock.json](configs/runtime-lock.json)。
本机ComfyUI位于 `.runtime/engines/ComfyUI-standard/`，依赖与缓存位于.runtime对应子目录。
保留现有Torch/Triton组合，不为整理目录升级环境。
本机GPU执行仍通过父工作区ROCm10 launcher、共享锁和ownership监控；GPU3排除。

本仓库是ComfyUI source extension。启动时加载本项目，保留
`--enable-triton-backend --disable-mmap --preview-method latent2rgb`，并传入
`--extra-model-paths-config <仓库>/configs/extra_model_paths.yaml`。
根 `__init__.py/nodes.py` 是框架发现入口，实际节点代码位于integrations。

## Attention

```bash
export H3_ATTENTION_BACKEND=reduced_error
export H3_ATTENTION_HEAD_CHUNK=7
export H3_ATTENTION_QUERY_TILE=128
export H3_ATTENTION_QUERY_GROUPS=2
```

| 名称 | 实现 |
|---|---|
| `reduced_error` | 默认、第一优先级；冻结INT8 QK / FP8 V编码、Ix HIP FP32 PV、直接BF16输出 |
| `int8_qk_fp8_pv` | Ix原始native INT8 QK / FP8 PV控制路径 |
| `fp8_qk_fp8_pv` | Ix SK1 native FP8 QK / FP8 PV控制路径 |
| `sageplus_triton` | 原AMD SagePlus Triton语义对照 |
| `torch_sdpa` | 显式PyTorch SDPA对照 |
| `aiter_6390_int8` | 固定PR #6390提交，gfx1201 HIP INT8 QK / FP8 PV；使用该PR自己的量化准备与consumer |
| `aiter_4188_bf16` | 固定PR #4188提交，gfx1201 FlyDSL BF16 dense attention |
| `aiter_4188_fp8` | 同一FlyDSL提交的FP8量化与attention，单独精度对照 |

reduced_error另支持显式BR512。后端失败直接报错，不自动切换；改变配置须重启两个rank。
单 query 组 BR256 使用显式 `H3_ATTENTION_QUERY_GROUPS=1`、
`H3_ATTENTION_QUERY_TILE=256` 和 `H3_ATTENTION_QGROUP_BUILD=<spill构建目录>`。
构建目录由 `tools/h3_attention_spill_build.py` 在锁定环境生成；扩展和源码 SHA 进入双 rank 校验。
该入口目前限定 Standard 的 `[1,37787,7,128]` 分组，保留原编码、FP32 PV 和 BF16 输出。
`H3_ATTENTION_QGROUP_AUDIT=1` 逐核心对照原核位模式，仅用于短验证；测速设为0。
两个AITER上游源码及依赖安装在本机 `.runtime/`，固定提交见[来源记录](configs/provenance/upstreams.json)。
这些后端的编码和数值边界不同于默认reduced_error；测速结果必须连同真实输入数值门和完整媒体验收一起阅读，不能据速度单独替换默认路径。
可在当前目录运行 `python3 tools/provision_aiter.py` 下载固定源码并在无GPU挂载的锁定容器里准备隔离依赖。
各路径的完整精度说明和回放记录在本地 `vibe/project/ATTENTION_BACKENDS.md`；公开资格范围见 [关键结果](docs/RESULTS.md)。

原生扩展在锁定环境、无GPU挂载时构建，缓存不进入Git：

```bash
python3 tools/build_attention.py ix
python3 tools/build_attention.py native-tile
```

本机已有Ix安装继续复用；重编译使用新的 `IXMX_RDNA4_INSTALL`，保留原安装。

## MLP SwiGLU

TP2 的 fc2 默认使用融合 BF16 SwiGLU，保留 SiLU 与乘 up 之后的两次 BF16 舍入，
后续 ConvRot256、行量化、INT8 GEMM 和输出边界沿用原实现。
`H3_FC2_FUSED_SWIGLU=0` 显式选择原 eager 路径；切换时重启两个 rank。
双 rank 会校验该开关一致。`H3_SWIGLU_AUDIT=1` 可逐调用核对激活的原始位模式，
它会增加同步，只用于短验证，测速时关闭。

## 可选 Cache-DiT

`Cache-DiT` 是**可选的近似计算模式，默认关闭**。TP2 的 50 个 DiT block
在前 `Fn` 个 block 后比较相邻采样步的残差；命中时复用此前完整计算的中段残差，
跳过中段 block，再执行尾部 `Bn` 个 block。两个 rank 使用全局、按元素数加权的
相对 L1 门共同决定命中。每次采样请求独立建缓存，结束即释放特征张量。
这是以潜在画质差异换取计算量的实验选项；阈值是中间残差门，**不是**最终视频或
音频误差上限。10 NFE 的 Standard 任务前 3 步完整计算、且限制连续命中 1 步，
可命中步数有限；当前没有经过完整同输入画质门和 1+3 热运行的速度结论。
开启后的成绩须单独标注 Cache-DiT，不与原无缓存 5090 Standard 成绩当作同等计算量比较。

先在锁定环境执行 `python3 tools/provision_cache_dit.py`，把固定版本的纯 Python
`cache-dit==1.5.3` 与 `diffusers==0.40.0` 放进忽略的 `.runtime/deps/`；
该工具不安装或改动 Torch、Triton、ROCm。启动 ComfyUI 前设置：

```bash
export H3_CACHE_DIT=1
export H3_CACHE_DIT_FN=8
export H3_CACHE_DIT_BN=0
export H3_CACHE_DIT_THRESHOLD=0.12
export H3_CACHE_DIT_WARMUP=3
export H3_CACHE_DIT_MAX_CONTINUOUS=1
```

这是本项目的保守初始配置，可按同输入画质和速度对照调整。关闭时设
`H3_CACHE_DIT=0` 或取消变量；关闭路径不导入可选依赖、不分配缓存特征。
切换配置后重启 ComfyUI 和两个 rank；缺少固定版本依赖会明确报错，不自动回退。
缓存只包住 DiT block 循环，不缓存 fresh TE、VAE、预览或 MP4 保存。
rank 日志记录逐步命中、相对 L1、执行与跳过的 block 数，便于判断实际收益。
固定真实 conditioning 的前 4 步功能回放中，阈值 0 的无命中路径与关闭模式
音视频 latent 逐位一致；阈值 0.12 在另一次前 4 步回放中没有命中。
阈值 0.50 仅用于验证命中分支：第 4 步跳过 42 个 block，视频/音频 latent
相对 L2 分别为 3.01%/2.06%。该阈值不是画质推荐值，四步 latent 差异
也不能代表完整视频观感或 10 NFE 的端到端速度。

## TE CPU 权重副本

`src/minimax_h3_amd_sp/te_cpu_backing.py` 提供显式 `CPUWeightReuse(clip.patcher)`，
在首次 GPU 加载前保留原始 CPU 参数、NVFP4 packed 数据和 scale 的引用。
调用 `set_enabled(True)` 后，TE 卸载时先等待设备完成，再恢复这些引用；
buffer 迁移和 ModelPatcher 清理由原路径执行。每次请求仍运行 fresh TE。
该对象仅用于 `torch.inference_mode()`、固定权重、无 LoRA 的单个模型实例；
可能额外保留约14.61 GiB CPU存储，不支持训练或修改参数。
不用时调用 `close()` 恢复原方法并释放引用。默认不创建此对象。真实 fresh TE 短 A/B/A 已通过 packed 权重、scale、
卸载后 CPU 字节和 conditioning 原始位检查；短测控制漂移较大，尚无稳定倍率，
同实例完整 Standard A/B/A 各1+3已通过原始输出与媒体/资源门；
TE offload中位8.342→0.317→3.155秒，总耗时中位189.882→186.549→180.831秒，
未证实稳定端到端收益，保持默认关闭。

`tools/h3_te_stage_probe.py` 提供消费方的低密度 TE 调用树诊断：
拆分旧模型卸载、TE加载、tokenize、forward及offload，记录CPU/缺页计数；
可复用预先创建的HIP事件对，在完整请求结束后读取。
CPU scope和current-stream事件跨度均不等于纯GPU layer时间，诊断不生成Standard成绩。
已完成三个完整上下文观察：旧DiT卸载占TE约49–51%，TE加载不足0.9秒。
后续第七次请求TE发生GPU page fault，精确原因未知，完整诊断缺少后置控制；
独立五循环驻留回放通过，另捕获到旧视频VAE卸载增加2.36秒。
这两类诊断范围分开记录于[关键结果](docs/RESULTS.md)，不证明稳定性、提速或观察器低开销。

独立 attention 寄存器压力实验使用 `tools/h3_attention_spill_build.py` 在无 GPU 的
锁定镜像中构建到新的私有 `--cache`，再在宿主运行：

```bash
python3 tools/h3_attention_spill_host.py \
  --build .runtime/cache/<new-build> --session attention-spill-<new-run>
```

该入口复用本地历史真实 TP2 QKV，核对同一编码、输入和 BF16 位模式后进行无 trace
交错回放，分别测7-head核心和28-head完整adapter。单query组候选在私有源码中增加
四处PV组访问的编译期保护；vendor和已安装扩展保持原样。实验不更改默认attention，
也不代表完整Standard或通信性能。

## RDNAScope 短诊断

本项目的短请求、GPU 预约和输出核对在 `tools/rdnascope_h3_host.py` 与
`tools/rdnascope_h3_short.py`。将 [配置模板](configs/rdnascope.example.toml)
复制成根目录 `.rdnascope.toml`，填写同一 ROCm SDK 对应的官方 Trace Processor
路径；RDNAScope 应以独立仓库安装并能在锁定容器中通过 `PYTHONPATH` 导入。

```bash
python3 tools/rdnascope_h3_host.py --session rdnascope-h3-tp2-<new-id>
```

脚本自动挑选两张空闲非 GPU3 卡，先执行 trace-off 一步 oracle，再由
RDNAScope 录制同输入的 `native_roi`，原始数据保留在忽略的 `runs/`。
该短请求只跑一次 NFE，不是 Standard 成绩。完整本地诊断与数值资格见
`vibe/research/RDNASCOPE_H3_TP2_SHORT_20261112.md`。

诊断 SwiGLU 开关时，可执行预热后的短 A/B/B/A 对照及两次 native 捕获：

```bash
python3 tools/rdnascope_h3_ab_host.py --session rdnascope-swiglu-ab-<new-id>
```

每个子请求包含一次首步预热和一次首步测量，均重新执行 TE；测量沿用
Standard 的输入形状、噪声与 10 步 schedule 的第一段，不解码音视频。
只在测量阶段采集：rank0 包含 sampler 加载，rank1 只包含 blocks，两者覆盖不同。
消费者 sidecar 提供 ROI；`rdnascope_record_safe.py` 临时处理 alpha 版统计目录大小时
SQLite journal 消失的竞态，仍保留超时、大小预算和符号链接检查，不修改工具仓库。
结果用于诊断；A/A 漂移、输出核对或覆盖不足时，不作为提速与精度验收。

## 文档和历史结果

仅 SwiGLU 融合已默认接入：本轮双 R9700 GPU5+6 Standard 中位 **220.019秒**，同卡对原路径控制 **205.910秒**；本轮没有证实端到端提速，资格范围见 [关键结果](docs/RESULTS.md)。

2026-11-11 整理基点的工程对照：双R9700热运行中位 **213.710秒**；同数学单卡对照 **880.113秒**。
单卡因外部占用中断后恢复，3次有效热运行来自2个会话，各先完整预热。AV latent和完整FP32解码像素逐元素相同；音频VAE解码waveform跨模式relative L2为1.51e-6，各模式内完全重复。
原C1精度门仍未通过，本轮为工程比较；完整资格范围见 [关键结果](docs/RESULTS.md)。
保留原数值门、fresh TE、10 NFE、预览与音频；新路径需真实输入回放后再进行完整Standard。
GitHub只展示 [关键结果](docs/RESULTS.md)，完整研究上下文从本地 `vibe/README.md` 进入。
日志、输出、完整参数表、迁移验收与历史文档均在vibe / archive保留，没有删除旧实验。
GitHub保留main代码与关键结果，完整研究记录保留在本地。

MultiGPU采用MIT；Ix / SageAttention保留Apache-2.0和NOTICE。
完整版本、许可证与移植范围见 [第三方说明](third_party/README.md)。
