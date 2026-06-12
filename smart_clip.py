# -*- coding: utf-8 -*-
"""
smart_clip.py — AI 自适应视频剪辑助手

像人类剪辑师一样工作：
  1. 自适应抽帧（粗扫 → 发现目标后加密 → 确认边界后快进），AI 自己决定节奏
  2. 可选 whisper 转录，用文字稿把剪切点精确到句子边界
  3. 自动剪出片段
  4. 按描述挑封面，给出备选 + 推荐理由
  5. 插入补充素材（画中画，自动压低素材音量当背景声）

用法示例：
  # 全自动模式：只给链接/文件，AI 自己选最值得剪的内容，出 1-3 分钟短视频 + 封面
  python smart_clip.py "https://www.bilibili.com/video/BVxxxx"

  # 指定要剪什么内容
  python smart_clip.py "https://youtu.be/xxxx" --task "讲书的片段"

  # 定位并剪出片段（核心功能）
  python smart_clip.py 直播.mp4 --task "脱不花在讲一本书，把讲书的片段找出来"

  # 定位后再浓缩成 1-3 分钟短视频（AI 挑精华段落拼接）
  python smart_clip.py 直播.mp4 --task "讲书的片段" --duration 1-3min

  # 顺便挑封面
  python smart_clip.py 直播.mp4 --task "讲书的片段" --cover "脱不花哭了的画面"

  # 已知时间点，直接剪
  python smart_clip.py 直播.mp4 --cut 00:45:00-01:25:00

  # 给成片插入手机拍摄素材（画中画，从成片第30秒开始）
  python smart_clip.py 成片.mp4 --pip 手机素材.mp4 --pip-at 00:00:30

  # 只挑封面
  python smart_clip.py 直播.mp4 --cover "拿着书讲解的画面"
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ============================================================
# 配置常量（key 放 config_local.py，不进 git；也支持环境变量）
# ============================================================
try:
    from config_local import AI_BASE_URL, AI_API_KEY, AI_MODEL, AI_FAST_MODEL
except ImportError:
    AI_BASE_URL   = os.environ.get("ANTHROPIC_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding")
    AI_API_KEY    = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    AI_MODEL      = os.environ.get("ANTHROPIC_MODEL", "doubao-seed-2.0-pro")     # 视觉主力
    AI_FAST_MODEL = os.environ.get("ANTHROPIC_SMALL_FAST_MODEL", "deepseek-v3.2")  # 文本快速分析

OUTPUT_ROOT   = Path.home() / "Downloads" / "smart_clip"

FRAME_WIDTH      = 640    # 送给 AI 看的帧宽度（够看清内容，省 token）
FRAMES_PER_CALL  = 10     # 每次 vision 调用最多带几张图
MAX_SCAN_ROUNDS  = 10     # agent 循环最大轮数
MAX_FRAMES_TOTAL = 300    # 总抽帧上限（防失控）
COVER_CANDIDATES = 16     # 封面初选张数上限

# ============================================================
# Prompt 模板
# ============================================================
OBSERVE_PROMPT = """你是视频剪辑助手，正在浏览一条视频的抽帧画面，目标任务：{TASK}

下面 {N} 张图按时间顺序排列，对应的时间戳依次是：{TIMESTAMPS}

请逐帧用一行话描述画面内容，重点关注与任务相关的线索（人物动作、手里的物品、屏幕文字、情绪表情等）。
输出格式（每帧一行，不要多余内容）：
[时间戳] 描述 | 与任务相关：是/否/可能"""

PLAN_PROMPT = """你是视频剪辑师，正在一条总时长 {DURATION} 的视频里定位目标片段。

任务：{TASK}

到目前为止的抽帧观察记录（按时间排序）：
{OBSERVATIONS}
{TRANSCRIPT_HINT}
请像人类剪辑师一样决定下一步：
- 还没发现目标 → 继续粗扫未看过的区域
- 发现疑似目标 → 在它附近加密抽帧，确认起点和终点
- 起止点都已确认 → 得出结论

只输出一个 JSON，不要其他文字：
继续扫描: {{"action": "scan", "start_sec": 数字, "end_sec": 数字, "interval_sec": 数字, "reason": "为什么这样扫"}}
得出结论: {{"action": "conclude", "clip_start_sec": 数字, "clip_end_sec": 数字, "reason": "判断依据"}}

注意：
- interval_sec 自己定（粗扫可以 120-300，确认边界可以 20-60）；单轮 (end-start)/interval 不要超过 30 帧
- 已观察过的时间点不会产生新信息；当目标边界两侧相邻已观察帧的间隔已经足够小（≤10 秒），不要再扫，直接 conclude：起点取最后一个"否"帧和第一个"是"帧的中点，终点同理"""

REFINE_PROMPT = """视频剪切点需要落在完整句子的边界上。任务：{TASK}

初定剪切区间：{START} 到 {END}

起点附近的转录文字（带秒数）：
{START_CONTEXT}

终点附近的转录文字（带秒数）：
{END_CONTEXT}

请选出最合适的剪切点：起点应落在目标内容第一句话开始之前的停顿处，终点落在最后一句话说完之后。
只输出 JSON：{{"start_sec": 数字, "end_sec": 数字, "reason": "简述"}}"""

AUTO_PLAN_PROMPT = """你是资深短视频编导。下面是一条长视频的完整转录稿（带秒数），
请通读全文，选出最值得剪成爆款短视频的一段内容。

转录稿：
{TRANSCRIPT}

评判标准：情绪浓度、观点锋利度、故事完整性、共鸣度、传播潜力。

只输出 JSON：
{{"chosen": {{"topic": "选题一句话", "start_sec": 数字, "end_sec": 数字,
            "cover_desc": "封面应该找什么画面", "why": "为什么这段最值得剪"}},
 "alternatives": [{{"topic": "备选选题", "start_sec": 数字, "end_sec": 数字, "why": "理由"}}]}}
备选给 1-2 个即可。start/end 要覆盖该话题完整的起止。"""

CONDENSE_PROMPT = """你是短视频剪辑师。下面是一段长视频片段的完整转录稿（带起止秒数），
需要从中精选内容，浓缩成一条 {MIN_SEC}-{MAX_SEC} 秒的短视频。

任务背景：{TASK}

转录稿：
{TRANSCRIPT}

剪辑要求：
1. 选 2-5 个不连续的精华段落，按时间顺序拼接（跳剪是短视频常规手法）
2. 第一段必须是钩子：最抓人、最有悬念或情绪的内容，让观众第一秒就停下来
3. 中间是核心干货/金句，结尾要有收束感（观点落地、情绪高点或行动号召）
4. 每段必须从完整句子开始、到完整句子结束，绝不能切在半句话上
5. 所有段落时长加起来必须在 {MIN_SEC}-{MAX_SEC} 秒之间

只输出 JSON：
{{"segments": [{{"start_sec": 数字, "end_sec": 数字, "why": "选这段的理由"}}], "summary": "成片内容概述"}}"""

COVER_SCORE_PROMPT = """你在为短视频挑封面，要求：{COVER_DESC}

下面 {N} 张候选帧，时间戳依次是：{TIMESTAMPS}

请给每帧打分（0-10，越符合要求越高，同时考虑清晰度、构图、表情张力），每帧一行：
[时间戳] 分数 | 一句话点评"""

COVER_FINAL_PROMPT = """你在为短视频挑封面，要求：{COVER_DESC}

这是初选出的 {N} 张高分帧，时间戳依次是：{TIMESTAMPS}

请从中选出 2 个备选方案，说明各自适合做封面的理由（情绪冲击力、构图、清晰度、观众共鸣），并给出你的最终推荐。
只输出 JSON：
{{"candidates": [{{"timestamp_sec": 数字, "reason": "理由"}}, {{"timestamp_sec": 数字, "reason": "理由"}}], "recommend_sec": 数字, "why": "推荐理由"}}"""

# ============================================================
# 工具函数
# ============================================================
def ensure_package(pkg, import_name=None):
    import importlib
    try:
        return importlib.import_module(import_name or pkg)
    except ImportError:
        print(f"  安装依赖 {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
        return importlib.import_module(import_name or pkg)


def format_timestamp(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def parse_timestamp(ts):
    parts = [float(p) for p in str(ts).split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def run_cmd(cmd, **kw):
    res = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if res.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(map(str, cmd))}\n{res.stderr[-2000:]}")
    return res.stdout


def get_client():
    if not AI_API_KEY:
        sys.exit("缺少 API key：复制 config_local.example.py 为 config_local.py 并填入你的 key")
    anthropic = ensure_package("anthropic")
    return anthropic.Anthropic(api_key=AI_API_KEY, base_url=AI_BASE_URL)


def ai_text(client, prompt, model=AI_FAST_MODEL):
    msg = client.messages.create(model=model, max_tokens=8192,
                                 messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


def ai_vision(client, prompt, image_paths):
    content = []
    for p in image_paths:
        img_b64 = base64.b64encode(Path(p).read_bytes()).decode()
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": img_b64}})
    content.append({"type": "text", "text": prompt})
    msg = client.messages.create(model=AI_MODEL, max_tokens=8192,
                                 messages=[{"role": "user", "content": content}])
    return msg.content[0].text


def parse_json_block(text):
    """取出回复里第一个完整的 JSON 对象（模型偶尔会输出多个，只取第一个）"""
    idx = text.find("{")
    if idx < 0:
        raise ValueError(f"AI 回复里没有 JSON：{text[:500]}")
    obj, _ = json.JSONDecoder().raw_decode(text[idx:])
    return obj


# ============================================================
# Step 函数
# ============================================================
def download_video(url, out_dir):
    """yt-dlp 下载视频（B站自动读 Chrome 登录态），返回本地文件路径"""
    yt_dlp = ensure_package("yt-dlp", "yt_dlp")
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out_dir / "%(title).80s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    if "bilibili.com" in url or "b23.tv" in url:
        ydl_opts["cookiesfrombrowser"] = ("chrome",)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
        if not path.exists():  # 合并后实际扩展名是 mp4
            path = path.with_suffix(".mp4")
    print(f"  标题: {info.get('title', '未知')}")
    print(f"  已下载: {path}")
    return path


def probe_video(video):
    out = run_cmd(["ffprobe", "-v", "quiet", "-print_format", "json",
                   "-show_format", "-show_streams", str(video)])
    info = json.loads(out)
    duration = float(info["format"]["duration"])
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    print(f"  时长 {format_timestamp(duration)} | {v['width']}x{v['height']} "
          f"| {float(info['format']['size']) / 1e9:.1f} GB")
    return duration


def extract_frame(video, sec, out_dir, width=FRAME_WIDTH):
    """抽单帧。-ss 放 -i 前面是关键帧快速定位，4 小时视频也秒级完成。
    片尾边界等抽不出帧的位置返回 None，调用方跳过即可。"""
    out = out_dir / f"f_{int(sec):06d}.jpg"
    if not out.exists():
        try:
            run_cmd(["ffmpeg", "-y", "-ss", str(sec), "-i", str(video),
                     "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "5", str(out)])
        except RuntimeError:
            return None
    return out


def observe_frames(client, video, secs, frames_dir, task, observations):
    """抽一批帧 → 分批送 vision 模型 → 把逐帧描述追加进观察记录"""
    secs = [s for s in secs if s not in {o[0] for o in observations}]
    for i in range(0, len(secs), FRAMES_PER_CALL):
        extracted = [(s, extract_frame(video, s, frames_dir)) for s in secs[i:i + FRAMES_PER_CALL]]
        batch = [s for s, p in extracted if p]
        paths = [p for _, p in extracted if p]
        if not paths:
            continue
        ts_list = ", ".join(format_timestamp(s) for s in batch)
        prompt = OBSERVE_PROMPT.format(TASK=task, N=len(batch), TIMESTAMPS=ts_list)
        reply = ai_vision(client, prompt, paths)
        for sec, line in zip(batch, [l for l in reply.splitlines() if l.strip().startswith("[")]):
            observations.append((sec, line.strip()))
            print(f"    {line.strip()}")
    observations.sort(key=lambda o: o[0])


def transcribe_audio(video, out_dir, model_size):
    """faster-whisper 转录，结果缓存成 json，重跑不重转"""
    cache = out_dir / "transcript.json"
    if cache.exists():
        print("  使用缓存的转录稿")
        return json.loads(cache.read_text())
    fw = ensure_package("faster-whisper", "faster_whisper")
    wav = out_dir / "audio.wav"
    if not wav.exists():
        print("  抽取音轨 ...")
        run_cmd(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1",
                 "-ar", "16000", str(wav)])
    print(f"  whisper({model_size}) 转录中（长视频可能要几十分钟）...")
    whisper = fw.WhisperModel(model_size, compute_type="int8")
    segments, _ = whisper.transcribe(str(wav), language="zh", vad_filter=True)
    result = [{"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip()}
              for s in segments]
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    wav.unlink(missing_ok=True)
    print(f"  转录完成，共 {len(result)} 句")
    return result


def transcript_excerpt(transcript, center_sec, radius=90):
    lines = [f"[{seg['start']:.0f}s] {seg['text']}" for seg in transcript
             if center_sec - radius <= seg["start"] <= center_sec + radius]
    return "\n".join(lines) or "（该区间无转录内容）"


def locate_segment(client, video, duration, task, frames_dir, transcript):
    """agent 循环：观察 → 决策 → 调整抽帧节奏，直到锁定起止点"""
    observations = []
    # 第一轮粗扫：全片均匀抽，间隔自适应（4 小时片约 3 分钟一帧）
    interval = max(min(duration / 80, 300), 10)
    plan = {"action": "scan", "start_sec": 0, "end_sec": duration,
            "interval_sec": interval, "reason": "首轮全片粗扫"}

    for round_no in range(1, MAX_SCAN_ROUNDS + 1):
        if plan["action"] == "conclude":
            break
        start, end = max(0, plan["start_sec"]), min(duration - 1, plan["end_sec"])
        step = max(plan["interval_sec"], (end - start) / 30, 5)
        secs = [round(s) for s in frange(start, end, step)]
        if len(observations) + len(secs) > MAX_FRAMES_TOTAL:
            secs = secs[:MAX_FRAMES_TOTAL - len(observations)]
        print(f"\n  第 {round_no} 轮 | {format_timestamp(start)}–{format_timestamp(end)} "
              f"每 {step:.0f}s 一帧（{len(secs)} 帧）| {plan['reason']}")
        seen_before = len(observations)
        observe_frames(client, video, secs, frames_dir, task, observations)
        stall_note = ""
        if len(observations) == seen_before:
            stall_note = ("\n（注意：上一轮请求的区间在该密度下没有产生任何新画面，"
                          "抽帧精度已达上限，请基于现有观察直接 conclude）")

        hint = ""
        if transcript:
            # 把疑似区域的转录稿喂给指挥员，画面+台词双重证据
            likely = [o[0] for o in observations if "相关：是" in o[1]]
            if likely:
                center = sum(likely) / len(likely)
                hint = f"\n疑似区域附近的转录文字：\n{transcript_excerpt(transcript, center, 300)}\n"
        obs_text = "\n".join(line for _, line in observations) + stall_note
        plan = parse_json_block(ai_text(client, PLAN_PROMPT.format(
            DURATION=format_timestamp(duration), TASK=task,
            OBSERVATIONS=obs_text, TRANSCRIPT_HINT=hint), model=AI_MODEL))

    if plan["action"] != "conclude":
        raise RuntimeError("达到最大扫描轮数仍未定位到目标片段，请换个任务描述试试")
    start, end = plan["clip_start_sec"], plan["clip_end_sec"]
    print(f"\n  ✅ 定位完成：{format_timestamp(start)} – {format_timestamp(end)}")
    print(f"  依据：{plan['reason']}")
    return float(start), float(end)


def frange(start, end, step):
    s = start
    while s <= end:
        yield s
        s += step


def refine_boundaries(client, task, start, end, transcript):
    """用转录稿把剪切点对齐到句子边界"""
    if not transcript:
        return start, end
    result = parse_json_block(ai_text(client, REFINE_PROMPT.format(
        TASK=task, START=format_timestamp(start), END=format_timestamp(end),
        START_CONTEXT=transcript_excerpt(transcript, start),
        END_CONTEXT=transcript_excerpt(transcript, end))))
    print(f"  句子边界微调：{format_timestamp(result['start_sec'])} – "
          f"{format_timestamp(result['end_sec'])}（{result['reason']}）")
    return float(result["start_sec"]), float(result["end_sec"])


def cut_clip(video, start, end, out_path):
    """重编码剪切，帧级精确"""
    run_cmd(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", str(video),
             "-ss", "0", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-c:a", "aac", "-b:a", "192k", str(out_path)])
    print(f"  ✅ 成片：{out_path}")


def parse_duration_range(text):
    """解析 '1-3min' / '60-180' / '90s' 之类的时长要求，返回 (最短秒, 最长秒)"""
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        raise ValueError(f"看不懂时长格式：{text}")
    unit = 60 if re.search(r"min|分", text) else 1
    lo = nums[0] * unit
    hi = nums[1] * unit if len(nums) > 1 else lo * 1.2
    return lo, hi


def auto_plan(client, transcript):
    """全自动模式：AI 通读转录稿，自己决定剪什么、封面找什么"""
    # 压缩成 ~30 秒一行，4 小时视频也能一次喂给模型
    blocks, cur, cur_start = [], [], 0
    for seg in transcript:
        if not cur:
            cur_start = seg["start"]
        cur.append(seg["text"])
        if seg["end"] - cur_start >= 30:
            blocks.append(f"[{cur_start:.0f}s] {''.join(cur)}")
            cur = []
    if cur:
        blocks.append(f"[{cur_start:.0f}s] {''.join(cur)}")
    plan = parse_json_block(ai_text(client, AUTO_PLAN_PROMPT.format(
        TRANSCRIPT="\n".join(blocks)), model=AI_MODEL))
    c = plan["chosen"]
    print(f"  🎯 AI 选题：{c['topic']}")
    print(f"     区间 {format_timestamp(c['start_sec'])}–{format_timestamp(c['end_sec'])}")
    print(f"     理由：{c['why']}")
    for alt in plan.get("alternatives", []):
        print(f"  备选：{alt['topic']} "
              f"[{format_timestamp(alt['start_sec'])}–{format_timestamp(alt['end_sec'])}] {alt['why']}")
    print(f"  （想换选题：重跑时加 --task \"备选选题描述\"）")
    return c


def condense_to_short(client, task, start, end, transcript, min_sec, max_sec):
    """从定位到的长片段里，让 AI 按"钩子-干货-收尾"挑精华段落"""
    lines = [f"[{seg['start']:.0f}-{seg['end']:.0f}] {seg['text']}"
             for seg in transcript if start <= seg["start"] < end]
    if not lines:
        raise RuntimeError("目标区间内没有转录内容，无法浓缩（是否用了 --no-transcribe？）")
    result = parse_json_block(ai_text(client, CONDENSE_PROMPT.format(
        TASK=task or "剪出精华短视频", TRANSCRIPT="\n".join(lines),
        MIN_SEC=int(min_sec), MAX_SEC=int(max_sec)), model=AI_MODEL))
    segments = sorted(result["segments"], key=lambda s: s["start_sec"])
    total = sum(s["end_sec"] - s["start_sec"] for s in segments)
    print(f"  精选 {len(segments)} 段，共 {format_timestamp(total)}：")
    for i, s in enumerate(segments, 1):
        print(f"    段{i} [{format_timestamp(s['start_sec'])}–{format_timestamp(s['end_sec'])}] {s['why']}")
    print(f"  成片概述：{result['summary']}")
    return segments


def cut_multi(video, segments, out_path):
    """多段剪切后拼接成一条成片"""
    tmp_dir = out_path.parent / "_segments"
    tmp_dir.mkdir(exist_ok=True)
    parts = []
    for i, s in enumerate(segments):
        part = tmp_dir / f"part_{i:02d}.mp4"
        run_cmd(["ffmpeg", "-y", "-ss", str(s["start_sec"]), "-to", str(s["end_sec"]),
                 "-i", str(video), "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-c:a", "aac", "-b:a", "192k", "-ar", "44100", str(part)])
        parts.append(part)
    concat_list = tmp_dir / "list.txt"
    concat_list.write_text("".join(f"file '{p.name}'\n" for p in parts))
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c", "copy", str(out_path)], cwd=tmp_dir)
    print(f"  ✅ 成片：{out_path}")


def pick_cover(client, video, start, end, cover_desc, frames_dir, out_dir):
    """密集抽帧 → 打分初选 → 终选给理由 → 导出原分辨率封面"""
    step = max((end - start) / 40, 2)
    secs = [round(s) for s in frange(start, end, step)]
    print(f"  封面海选：{len(secs)} 帧")
    scored = []
    for i in range(0, len(secs), FRAMES_PER_CALL):
        extracted = [(s, extract_frame(video, s, frames_dir)) for s in secs[i:i + FRAMES_PER_CALL]]
        batch = [s for s, p in extracted if p]
        paths = [p for _, p in extracted if p]
        if not paths:
            continue
        reply = ai_vision(client, COVER_SCORE_PROMPT.format(
            COVER_DESC=cover_desc, N=len(batch),
            TIMESTAMPS=", ".join(format_timestamp(s) for s in batch)), paths)
        for sec, line in zip(batch, [l for l in reply.splitlines() if l.strip().startswith("[")]):
            m = re.search(r"\]\s*(\d+(?:\.\d+)?)", line)
            if m:
                scored.append((float(m.group(1)), sec, line.strip()))
    scored.sort(reverse=True)
    finalists = scored[:min(COVER_CANDIDATES, len(scored))][:FRAMES_PER_CALL]
    if not finalists:
        print("  ⚠️ 没找到符合描述的画面")
        return
    f_secs = [sec for _, sec, _ in finalists]
    f_paths = [extract_frame(video, s, frames_dir) for s in f_secs]
    result = parse_json_block(ai_vision(client, COVER_FINAL_PROMPT.format(
        COVER_DESC=cover_desc, N=len(f_secs),
        TIMESTAMPS=", ".join(format_timestamp(s) for s in f_secs)), f_paths))

    covers_dir = out_dir / "covers"
    covers_dir.mkdir(exist_ok=True)
    print("\n  封面方案：")
    for i, c in enumerate(result["candidates"], 1):
        sec = c["timestamp_sec"]
        out = covers_dir / f"cover_{i}_{format_timestamp(sec).replace(':', '')}.jpg"
        run_cmd(["ffmpeg", "-y", "-ss", str(sec), "-i", str(video),
                 "-frames:v", "1", "-q:v", "2", str(out)])
        star = " ⭐推荐" if abs(sec - result["recommend_sec"]) < 1 else ""
        print(f"    方案{i}{star} [{format_timestamp(sec)}] {c['reason']}\n      → {out}")
    print(f"  推荐理由：{result['why']}")
    (covers_dir / "covers.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))


def insert_pip(client, main_video, pip_video, at_sec, out_path):
    """画中画插入：素材缩小放右上角，音量压成背景声不盖主音轨"""
    pip_dur = float(json.loads(run_cmd(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
         str(pip_video)]))["format"]["duration"])
    fc = (
        f"[1:v]scale=iw*0.35:-2[pip];"
        f"[0:v][pip]overlay=W-w-40:40:enable='between(t,{at_sec},{at_sec + pip_dur})'[v];"
        f"[1:a]volume=0.15,adelay={int(at_sec * 1000)}|{int(at_sec * 1000)}[bga];"
        f"[0:a][bga]amix=inputs=2:duration=first:normalize=0[a]"
    )
    run_cmd(["ffmpeg", "-y", "-i", str(main_video), "-i", str(pip_video),
             "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-c:a", "aac", "-b:a", "192k", str(out_path)])
    print(f"  ✅ 画中画完成（素材音量已压至 15% 作背景声）：{out_path}")


# ============================================================
# 主流程
# ============================================================
def run(video, task=None, cover=None, cut=None, duration=None, pip=None,
        pip_at="0", transcribe=True, whisper_model="medium"):
    if re.match(r"https?://", str(video)):
        print(f"\n[0/5] 下载视频")
        video = download_video(str(video), OUTPUT_ROOT / "downloads")
    else:
        video = Path(video).expanduser()
    if not video.exists():
        sys.exit(f"找不到文件：{video}")
    out_dir = OUTPUT_ROOT / video.stem
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    client = get_client()

    print(f"\n[1/5] 读取视频信息")
    total_dur = probe_video(video)

    auto_mode = not (task or cut or cover or pip)
    if auto_mode:
        print("  （全自动模式：AI 自己选内容、定时长、挑封面）")
        transcribe = True
        duration = duration or "1-3min"

    transcript = None
    if transcribe and (task or cut or auto_mode):
        print(f"\n[2/5] 转录音频")
        transcript = transcribe_audio(video, out_dir, whisper_model)

    start = end = None
    if auto_mode:
        print(f"\n[3/5] AI 通读全片选题")
        chosen = auto_plan(client, transcript)
        task = chosen["topic"]
        cover = chosen["cover_desc"]
        start, end = float(chosen["start_sec"]), float(chosen["end_sec"])
    elif cut:
        s, e = cut.split("-")
        start, end = parse_timestamp(s), parse_timestamp(e)
        print(f"\n[3/5] 使用指定区间 {format_timestamp(start)} – {format_timestamp(end)}")
    elif task:
        print(f"\n[3/5] AI 定位目标片段")
        start, end = locate_segment(client, video, total_dur, task, frames_dir, transcript)
        start, end = refine_boundaries(client, task, start, end, transcript)

    clip_path = None
    if start is not None:
        print(f"\n[4/5] 剪切片段")
        min_sec = max_sec = None
        if duration:
            min_sec, max_sec = parse_duration_range(duration)
        if max_sec and (end - start) > max_sec:
            if transcript:
                print(f"  定位片段 {format_timestamp(end - start)}，浓缩至 "
                      f"{int(min_sec)}-{int(max_sec)} 秒")
                segments = condense_to_short(client, task, start, end,
                                             transcript, min_sec, max_sec)
                clip_path = out_dir / f"short_{format_timestamp(start).replace(':', '')}.mp4"
                cut_multi(video, segments, clip_path)
            else:
                print(f"  ⚠️ 没有转录稿无法挑精华，直接截取前 {int(max_sec)} 秒"
                      f"（去掉 --no-transcribe 可启用 AI 精选）")
                clip_path = out_dir / f"clip_{format_timestamp(start).replace(':', '')}.mp4"
                cut_clip(video, start, start + max_sec, clip_path)
        else:
            clip_path = out_dir / f"clip_{format_timestamp(start).replace(':', '')}.mp4"
            cut_clip(video, start, end, clip_path)

    if cover:
        print(f"\n[5/5] 挑选封面")
        c_start, c_end = (start, end) if start is not None else (0, total_dur)
        pick_cover(client, video, c_start, c_end, cover, frames_dir, out_dir)

    if pip:
        print(f"\n[附加] 插入画中画素材")
        target = clip_path or video
        insert_pip(client, target, Path(pip).expanduser(), parse_timestamp(pip_at),
                   out_dir / f"{target.stem}_pip.mp4")

    print(f"\n全部完成，产出目录：{out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI 自适应视频剪辑助手")
    ap.add_argument("video", nargs="?", help="视频文件路径，或 YouTube/B站 链接（不填则运行时提示输入）")
    ap.add_argument("--task", help="要找的内容，如：脱不花讲一本书的片段")
    ap.add_argument("--cover", help="封面画面描述，如：哭了的画面")
    ap.add_argument("--cut", help="跳过定位直接剪，格式 HH:MM:SS-HH:MM:SS")
    ap.add_argument("--duration", help="成片目标时长，如 1-3min / 90s，超长时 AI 自动挑精华拼接")
    ap.add_argument("--pip", help="画中画素材文件")
    ap.add_argument("--pip-at", default="0", help="画中画插入时间点，默认开头")
    ap.add_argument("--no-transcribe", action="store_true", help="跳过音频转录（只靠画面定位）")
    ap.add_argument("--whisper-model", default="medium", help="whisper 模型：small/medium/large")
    args = ap.parse_args()
    if not args.video:
        print("🎬 AI 视频剪辑助手（全自动模式：自己选内容、剪 1-3 分钟短视频、配封面）")
        args.video = input("请粘贴视频链接（B站/YouTube）或本地文件路径：").strip().strip("'\"")
        if not args.video:
            sys.exit("没有输入，退出")
        extra = input("有特别想剪的内容吗？（直接回车 = AI 自己选）：").strip()
        if extra:
            args.task = extra
    run(args.video, task=args.task, cover=args.cover, cut=args.cut,
        duration=args.duration, pip=args.pip, pip_at=args.pip_at,
        transcribe=not args.no_transcribe, whisper_model=args.whisper_model)
