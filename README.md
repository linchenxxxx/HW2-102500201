# 第二次挑战：SGLang 前缀缓存与请求流程

姓名：林尘汐。`report.pdf` 包含任务一的实测对照、因果注意力分析及任务二的源码流程图；第四、第五点已留空，其中第五点第（2）（3）小题须由本人独立撰写。`AI 使用说明情况（第二次挑战）.pdf` 是待本人填写的表单。

## 实验环境

| 项目 | 本次运行 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB，驱动 595.91.07 |
| 模型 | Qwen/Qwen3-0.6B，Hugging Face 缓存 revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| SGLang | 0.5.14 |
| Ray | 2.56.0；仅作为统一软件环境安装，单卡服务 `use_ray=false` |
| Python | 3.10，环境 `/home/xi/first-challenge-sglang-py310` |
| 推理设置 | float16，TP=1，context-length=4096，mem-fraction-static=0.72，max-running-requests=8，RadixCache 开启，page-size=1 |

在兼容 CUDA 的主机上可以新建 Python 3.10 环境，并安装固定版本：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install 'sglang[all]==0.5.14' 'ray==2.56.0' aiohttp reportlab
```

模型可提前用 Hugging Face 下载，也可在可访问 Hugging Face 的网络下直接以模型 ID 启动。GPU 驱动、CUDA 与 PyTorch 应匹配主机。

## 启动与回放

本次实际使用本机缓存快照，禁用重叠调度以对应 `Scheduler.event_loop_normal`，但保留 RadixCache：

```bash
PATH='/home/xi/first-challenge-sglang-py310/bin':$PATH HF_HUB_OFFLINE=1 \
/home/xi/first-challenge-sglang-py310/bin/python -m sglang.launch_server \
  --model-path '/home/xi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca' \
  --host 127.0.0.1 --port 30000 --dtype float16 --tp-size 1 \
  --context-length 4096 --mem-fraction-static 0.72 --max-running-requests 8 \
  --disable-cuda-graph --disable-overlap-schedule
```

在另一终端、仓库根目录运行：

```bash
/home/xi/first-challenge-sglang-py310/bin/python src/target1/benchmark.py
/home/xi/first-challenge-sglang-py310/bin/python src/target1/verify_results.py
python src/build_documents.py
```

跨机器回放时，把以上绝对路径换为自己的环境与模型路径；脚本也支持 `--base-url` 和 `--results`。若保留本次原始结果，可用 `--results results/target1/run-2` 进行追加实验。重新生成主报告时，`src/build_documents.py` 默认读取 `results/target1/comparison.json`。

## 脚本与结果对应

`src/target1/benchmark.py` 用固定随机种子构造两组各 32 条 2112-token 请求，最大并发 8。它先完成短请求服务预热；每组前等待上组完成、调用并校验 `/flush_cache`；共享组额外发送 1 条相同 2048-token 前缀的预热请求。所有请求使用原生 `/generate` 的 `input_ids` 和 SSE 流，固定 `temperature=0`、`max_new_tokens=16`、`ignore_eos=true`、`sampling_seed=2026`。它保存逐请求结果并以原始值计算主表统计。

`src/build_documents.py` 从汇总 JSON 生成 `report.pdf` 和留空的 AI 使用说明 PDF。源码流程中的行号取自本机 SGLang 0.5.14 安装包，均位于 `sglang/srt/` 下；[SGLang v0.5.14 源码](https://github.com/sgl-project/sglang/tree/v0.5.14/python/sglang/srt) 可用于核对。

`src/target1/verify_results.py` 核验 64 条请求的长度、参数、输入哈希、成功状态、逐条缓存数和汇总值与原始数据的一致性。

| 报告内容 | 原始文件 |
|---|---|
| 主表分散前缀列 | `results/target1/dispersed_prefix/summary.json`，由同目录 `responses.jsonl` 汇总 |
| 主表共享前缀列 | `results/target1/shared_prefix/summary.json`，由同目录 `responses.jsonl` 汇总 |
| 两组对照 | `results/target1/comparison.json` |
| 输入长度、参数与顺序 | 两组各自的 `requests.jsonl` |
| 清缓存与预热 | 两组 `flush.json`、共享组 `prewarm.json`、根目录 `service_warmup.json` |
| 设备与版本 | `results/target1/run_metadata.json` |

逐请求 `cached_tokens` 和 `prompt_tokens` 来自 SGLang 响应 `meta_info`。缓存命中率按两者总和之比计算；实际 Prefill token 为 `sum(prompt_tokens - cached_tokens)`。TTFT、TPOT 与端到端延迟的定义写在报告第一页。吞吐量以输出 token 总数除以本组 32 条请求的完整墙钟时间。组间绝对延迟受 GPU 负载、JIT 热身、调度与主机状态影响；建议关注相同条件下的缓存命中与 Prefill 差异。
