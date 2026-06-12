# 🎬 smart-clip — AI 自适应视频剪辑助手

丢给它一个 B站/YouTube 链接（或本地视频），AI 自己看完整条视频，**自己决定剪什么**，产出 1-3 分钟短视频 + 封面备选（附推荐理由）。

像人类剪辑师一样工作：

- **自适应抽帧**：先 3 分钟一帧粗扫全片 → 发现目标后加密到 1 分钟一帧 → 确认边界后快进扫尾。节奏由 AI 自己决定，不是固定脚本
- **双重证据定位**：画面（豆包 Vision 逐帧理解）+ 台词（whisper 转录带时间戳），剪切点精确对齐到完整句子边界
- **AI 自主选题**：通读全片转录稿，按情绪浓度/观点锋利度/传播潜力选出最值得剪的内容，并给出备选选题和理由
- **智能浓缩**：定位到的片段超长时，按「钩子 → 干货 → 收尾」结构挑 2-5 段精华跳剪拼接
- **封面方案**：按描述海选画面 → 打分 → 给出 2 个备选方案 + 各自适合的理由 + 最终推荐
- **画中画**：插入补充素材时自动缩小画面、压低素材音量当背景声，不盖主音轨

## 安装

```bash
brew install ffmpeg                      # 系统依赖
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config_local.example.py config_local.py   # 填入你的火山方舟 Ark API key
```

## 用法

### 全自动模式（推荐）

```bash
.venv/bin/python smart_clip.py
# 按提示粘贴链接，回车，等着收片
```

或直接带参数：

```bash
.venv/bin/python smart_clip.py "https://www.bilibili.com/video/BVxxxx"
```

AI 会自动：下载 → 转录 → 通读选题 → 剪 1-3 分钟成片 → 挑封面。

### 指定内容

```bash
# 告诉它剪什么
.venv/bin/python smart_clip.py 直播.mp4 --task "脱不花在讲一本书，把讲书的片段找出来"

# 加封面要求和目标时长
.venv/bin/python smart_clip.py 直播.mp4 --task "讲书的片段" --duration 1-3min --cover "情绪最强的画面"

# 已知时间点直接剪
.venv/bin/python smart_clip.py 直播.mp4 --cut 00:45:00-01:25:00

# 插入手机拍摄素材（画中画，素材音量自动压成背景声）
.venv/bin/python smart_clip.py 成片.mp4 --pip 手机素材.mp4 --pip-at 00:00:30
```

### 全部参数

| 参数 | 说明 |
|------|------|
| `video` | 视频文件路径或 B站/YouTube 链接，不填则交互式提示输入 |
| `--task` | 要找的内容描述，不填则 AI 自己选题 |
| `--duration` | 成片目标时长，如 `1-3min` / `90s`，超长自动挑精华拼接 |
| `--cover` | 封面画面描述，全自动模式下 AI 自己定 |
| `--cut` | 跳过定位直接剪，格式 `HH:MM:SS-HH:MM:SS` |
| `--pip` / `--pip-at` | 画中画素材文件 / 插入时间点 |
| `--no-transcribe` | 跳过音频转录（只靠画面定位，无法浓缩） |
| `--whisper-model` | whisper 模型：`small` / `medium`(默认) / `large` |

## 产出

统一放在 `~/Downloads/smart_clip/<视频名>/`：

```
clip_*.mp4 / short_*.mp4   成片
covers/                    封面备选 + covers.json（理由）
transcript.json            转录稿缓存（重跑不重转）
frames/                    抽过的帧缓存（重跑不重抽）
```

## 说明

- B 站链接自动读取 Chrome 登录态（`cookiesfrombrowser`），可下载登录可见的清晰度
- 4 小时直播首轮定位约 15-25 次模型调用；whisper 转录较耗时但有缓存
- API 走火山方舟（豆包），用 anthropic SDK + 自定义 base_url 调用
