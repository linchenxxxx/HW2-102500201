#!/usr/bin/env python3
"""Build the four-page report and the intentionally blank AI-use form."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
W, H = A4
LEFT = 47
RIGHT = W - 47
WIDTH = RIGHT - LEFT
INK = colors.HexColor("#17324A")
BLUE = colors.HexColor("#12658A")
PALE = colors.HexColor("#EAF3F7")
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
FONT = "STSong-Light"


def text(c, x, y, value, size=10, color=INK):
    c.setFillColor(color)
    c.setFont(FONT, size)
    c.drawString(x, y, value)


def wrap(value, size, width):
    lines = []
    for paragraph in value.split("\n"):
        line = ""
        for char in paragraph:
            if pdfmetrics.stringWidth(line + char, FONT, size) > width and line:
                lines.append(line)
                line = char
            else:
                line += char
        lines.append(line)
    return lines


def paragraph(c, y, value, size=10, leading=16, indent=0):
    for line in wrap(value, size, WIDTH - indent):
        text(c, LEFT + indent, y, line, size)
        y -= leading
    return y - 5


def heading(c, y, value):
    text(c, LEFT, y, value, 13, BLUE)
    c.setStrokeColor(colors.HexColor("#C4D6E0"))
    c.line(LEFT, y - 7, RIGHT, y - 7)
    return y - 28


def page(c, n, subtitle):
    c.setStrokeColor(colors.HexColor("#DBE5EB"))
    c.line(LEFT, 40, RIGHT, 40)
    text(c, LEFT, 27, "第二次挑战：林尘汐", 8)
    text(c, RIGHT - 72, 27, f"{subtitle}  {n}/4", 8)


def table(c, y, headers, rows, widths, row_h=29, font_size=9):
    x = LEFT
    c.setFillColor(PALE)
    c.rect(LEFT, y - row_h, WIDTH, row_h, fill=1, stroke=0)
    for value, width in zip(headers, widths):
        text(c, x + 5, y - 19, str(value), font_size, BLUE)
        x += width
    y -= row_h
    for index, row in enumerate(rows):
        if index % 2:
            c.setFillColor(colors.HexColor("#F7FAFC"))
            c.rect(LEFT, y - row_h, WIDTH, row_h, fill=1, stroke=0)
        x = LEFT
        for value, width in zip(row, widths):
            text(c, x + 5, y - 19, str(value), font_size)
            x += width
        y -= row_h
    c.setStrokeColor(colors.HexColor("#C4D6E0"))
    c.line(LEFT, y, RIGHT, y)
    return y - 15


def ms(value):
    return f"{value * 1000:.1f}"


def build_report():
    summaries = json.loads((ROOT / "results/target1/comparison.json").read_text(encoding="utf-8"))
    d, s = summaries
    c = canvas.Canvas(str(ROOT / "report.pdf"), pagesize=A4)
    c.setTitle("第二次挑战：SGLang 前缀缓存与请求流程")
    c.setAuthor("林尘汐")

    # Page 1: design and metric definitions.
    text(c, LEFT, 790, "第二次挑战：前缀缓存与请求流程", 19, BLUE)
    text(c, LEFT, 766, "SGLang 0.5.14 | Ray 2.56.0 | Qwen/Qwen3-0.6B", 10)
    y = heading(c, 729, "任务一：实验设计")
    y = paragraph(c, y, "设备为 NVIDIA GeForce RTX 5060 Laptop GPU（8151 MiB）；单卡 TP=1，float16，context-length=4096，mem-fraction-static=0.72，max-running-requests=8。服务日志显示 RadixCache 已启用，page-size=1；使用原生 POST /generate、input_ids 与 SSE 流式响应。Ray 2.56.0 安装在同一环境，本次单卡服务的 use_ray=false。")
    y = table(c, y, ["组别", "输入结构", "测量前准备"], [
        ["共享前缀", "2048 共同 + 64 独立", "flush 后先发 1 条同前缀预热"],
        ["分散前缀", "2112 全部独立，首 token 各异", "flush 后直接测量"],
    ], [91, 176, 234], row_h=29, font_size=8.8)
    y = paragraph(c, y, "两组各 32 条测量请求，最大并发均为 8；输入均为 2112 token。temperature=0、max_new_tokens=16、ignore_eos=true、sampling_seed=2026。请求用固定随机种子 2026 构造，组内顺序固定。共享组预热请求与测量请求有相同的 2048 token 前缀，64 token 后缀不同；预热请求不计入统计。")
    y = paragraph(c, y, "先用 4 token 短请求完成服务预热。每组测量前，上一组请求全部结束后调用 POST /flush_cache?timeout=30；脚本校验 HTTP 200 与“Cache flushed.”，然后才继续。运行顺序为分散组、共享组；每组计时窗口从提交 32 条请求开始，到最后一条流结束为止。")
    y = heading(c, y - 5, "指标口径与可追溯数据")
    y = paragraph(c, y, "缓存命中率 = sum(cached_tokens) / sum(prompt_tokens)；实际 Prefill token = sum(prompt_tokens - cached_tokens)。吞吐量 = 测量组总输出 token / 组墙钟时间。TTFT 从发起请求计到首个含生成 token 的 SSE 事件；端到端延迟计到 [DONE]；TPOT = (端到端延迟 - TTFT) / (输出 token 数 - 1)。p50、p95 使用对逐请求值做线性插值。")
    y = paragraph(c, y, "results/target1/ 各组的 requests.jsonl 保存 input_ids 和参数；responses.jsonl 保存每条请求的 token 数、命中数、计时及状态；summary.json 与 comparison.json 对应下页主表。flush.json、prewarm.json、service_warmup.json 可核验实验顺序。")
    page(c, 1, "设计")
    c.showPage()

    # Page 2: all required metrics and theory.
    text(c, LEFT, 790, "任务一：结果与分析", 17, BLUE)
    y = 758
    y = table(c, y, ["指标", "分散前缀", "共享前缀"], [
        ["成功率", "32/32 (100%)", "32/32 (100%)"],
        ["输出吞吐量 (token/s)", f"{d['throughput_output_tokens_per_s']:.2f}", f"{s['throughput_output_tokens_per_s']:.2f}"],
        ["缓存命中率", f"{d['cache_hit_rate']*100:.2f}%", f"{s['cache_hit_rate']*100:.2f}%"],
        ["实际 Prefill token", f"{d['total_prefill_tokens']:,}", f"{s['total_prefill_tokens']:,}"],
        ["TTFT p50 / p95 (ms)", f"{ms(d['ttft_s_p50'])} / {ms(d['ttft_s_p95'])}", f"{ms(s['ttft_s_p50'])} / {ms(s['ttft_s_p95'])}"],
        ["TPOT p50 / p95 (ms)", f"{ms(d['tpot_s_p50'])} / {ms(d['tpot_s_p95'])}", f"{ms(s['tpot_s_p50'])} / {ms(s['tpot_s_p95'])}"],
        ["端到端 p50 / p95 (ms)", f"{ms(d['e2e_s_p50'])} / {ms(d['e2e_s_p95'])}", f"{ms(s['e2e_s_p50'])} / {ms(s['e2e_s_p95'])}"],
    ], [236, 132, 133], row_h=32, font_size=9)
    y = paragraph(c, y, f"两组输入合计均为 {d['total_prompt_tokens']:,} token，输出合计均为 {d['total_completion_tokens']} token；共享组命中 {s['total_cached_tokens']:,} token（每条 2048），仅需执行 32×64={s['total_prefill_tokens']:,} token 的 Prefill。分散组首 token 各异，命中 0。吞吐量从 {d['throughput_output_tokens_per_s']:.2f} 增至 {s['throughput_output_tokens_per_s']:.2f} token/s，主要来自共享组大幅缩短的 Prefill。")
    y = heading(c, y - 4, "为什么相同前缀可复用")
    y = paragraph(c, y, "因果自注意力为 Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k) + causal_mask) V，且每层位置 i 有 K_i = h_i W_K、V_i = h_i W_V。因果掩码使位置 i 的隐状态只依赖位置 1 到 i 的 token；同一模型、同一 token 序列及位置的前缀，其逐层 h_i 与 K/V 相同，后续独立后缀不会改变已算好的前缀 K/V。RadixCache 命中后提供相应 KV 索引，Prefill 只计算未命中的 64 token 后缀；新生成的 16 token 仍需逐步 Decode。")
    y = paragraph(c, y, f"TTFT 包含排队与 Prefill，因此共享组 p50 从 {ms(d['ttft_s_p50'])} ms 降至 {ms(s['ttft_s_p50'])} ms，p95 从 {ms(d['ttft_s_p95'])} ms 降至 {ms(s['ttft_s_p95'])} ms。TPOT 所对应的 Decode 计算本身没有被前缀缓存免除；本次 TPOT 也下降（p50 {ms(d['tpot_s_p50'])}→{ms(s['tpot_s_p50'])} ms），可由较短 Prefill 减少 GPU 争用、排队及调度干扰解释，不能据此推断单个 Decode token 的算术量减少。端到端延迟同时受 TTFT 与后续生成影响。")
    page(c, 2, "结果")
    c.showPage()

    # Page 3: main request path, with function and file at each stage.
    text(c, LEFT, 790, "任务二：/generate 请求主流程", 17, BLUE)
    boxes = [
        ("HTTP 接收 / SSE", "entrypoints/http_server.py: generate_request（约 769 行）"),
        ("请求解析与派发", "managers/tokenizer_manager.py: generate_request / _send_one_request（576 / 1319 行）"),
        ("进入等待队列", "managers/scheduler.py: event_loop_normal / handle_generate_request / _add_request_to_queue（1505 / 1998 / 2258 行）"),
        ("前缀匹配与组批", "schedule_batch.py: Req.init_next_round_input（1123）；schedule_policy.py: match_prefix_for_req（85）；radix_cache.py: match_prefix（358）"),
        ("Prefill 预算 / 执行", "managers/schedule_policy.py: PrefillAdder.add_one_req（858 行）；schedule_batch.py: prepare_for_extend（2008 行）"),
        ("模型执行：Prefill / Decode", "managers/scheduler.py: run_batch（3145 行）；managers/tp_worker.py: forward_batch_generation（482 行）"),
        ("缓存更新与流式输出", "mem_cache/radix_cache.py: cache_finished_req / cache_unfinished_req（438 / 485 行）→ tokenizer_manager.py: _wait_one_response（1425 行）"),
    ]
    y = 756
    for index, (label, location) in enumerate(boxes):
        c.setFillColor(PALE if index % 2 == 0 else colors.HexColor("#F6F9FB"))
        c.roundRect(LEFT, y - 49, WIDTH, 45, 5, fill=1, stroke=0)
        text(c, LEFT + 10, y - 19, label, 10, BLUE)
        for line_index, line in enumerate(wrap(location, 8, WIDTH - 155)[:2]):
            text(c, LEFT + 148, y - 18 - line_index * 11, line, 8)
        y -= 54
        if index < len(boxes) - 1:
            text(c, LEFT + WIDTH / 2 - 4, y + 3, "↓", 11, BLUE)
            y -= 9
    y = heading(c, y - 17, "主流程说明（限一页）")
    y = paragraph(c, y, "TokenizerManager 将 input_ids 组成 TokenizedGenerateReqInput 后发给 Scheduler；event_loop_normal 从接收器取请求，process_input_requests → handle_generate_request 构造 Req，_add_request_to_queue 将普通请求追加至 waiting_queue。", 9.3, 14)
    y = paragraph(c, y, "get_new_batch_prefill 读取等待队列。当前 FCFS 路径中，Req.init_next_round_input 直接调用 RadixCache.match_prefix，返回已缓存前缀的 KV 索引；schedule_policy.match_prefix_for_req 是缓存感知调度的辅助入口。extend_input_len = 全部待处理 token - 命中前缀长度。PrefillAdder 按实际延展长度计预算，ScheduleBatch.prepare_for_extend 仅打包未命中的 token。", 9.3, 14)
    y = paragraph(c, y, "run_batch 调用 model worker 执行 Prefill 与后续 Decode。未结束请求经 maybe_cache_unfinished_req、结束请求经 release_kv_cache，分别调用 RadixCache.cache_unfinished_req / cache_finished_req，把已提交 token 的 KV 索引插入树并管理引用。Scheduler 输出经 TokenizerManager._wait_one_response 逐次 yield，由 HTTP 层编码成 data: ... 的 SSE 事件，最后发送 [DONE]。", 9.3, 14)
    y = paragraph(c, y, "上述相对路径以 SGLang 0.5.14 安装包的 sglang/srt/ 为根；函数名与行号可在该版本源码中核对。", 8.8, 13)
    page(c, 3, "流程")
    c.showPage()

    # Page 4: prompts deliberately left for the student.
    text(c, LEFT, 790, "第四、第五点：请本人填写", 17, BLUE)
    y = heading(c, 750, "四：AI 使用说明情况")
    y = paragraph(c, y, "请在随附的《AI 使用说明情况（第二次挑战）.pdf》中填写：使用的模型、提示词、AI 如何帮助学习、是否产生误导及如何核验。此处留待本人填写。")
    y = heading(c, y - 15, "五：作业感受")
    prompts = [
        "（1）如何完成第二次挑战：AI 使用方式与工作迭代等。",
        "（2）最困难的部分及克服方法。（题目要求禁止 AI 辅助撰写）",
        "（3）从 0 到 1 的科研工作需要哪些研究、对今后科研的启发。（题目要求禁止 AI 辅助撰写）",
    ]
    for prompt in prompts:
        y = paragraph(c, y, prompt, 10)
        for _ in range(4):
            c.setStrokeColor(colors.HexColor("#CCD9E0"))
            c.line(LEFT + 13, y - 12, RIGHT, y - 12)
            y -= 27
        y -= 16
    page(c, 4, "待填")
    c.save()


def build_ai_form():
    target = ROOT / "AI 使用说明情况（第二次挑战）.pdf"
    c = canvas.Canvas(str(target), pagesize=A4)
    c.setTitle("AI 使用说明情况（第二次挑战）—待本人填写")
    text(c, LEFT, 790, "AI 使用说明情况（第二次挑战）", 18, BLUE)
    text(c, LEFT, 763, "姓名：林尘汐                         以下内容请本人填写", 10)
    y = 730
    for label, lines in [
        ("1. 使用的 AI 模型与版本", 3),
        ("2. 使用的 AI 提示词（可附完整记录）", 7),
        ("3. AI 如何帮助学习", 6),
        ("4. AI 是否造成误导；如何发现、核验和纠正", 6),
    ]:
        y = heading(c, y, label)
        for _ in range(lines):
            c.setStrokeColor(colors.HexColor("#CCD9E0"))
            c.line(LEFT + 10, y - 6, RIGHT, y - 6)
            y -= 23
        y -= 16
    c.save()


if __name__ == "__main__":
    build_report()
    build_ai_form()
