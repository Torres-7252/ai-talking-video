# AI数字人口播视频生成器

基于 RTX 4060 的本地中文 AI 口播视频生成流水线。

## 技术栈

| 模块 | 技术 | 功能 |
|------|------|------|
| AI声音 | GPT-SoVITS | 文案 → 中文语音 |
| 数字人 | MuseTalk 1.5 | 音频驱动人物嘴型同步 |
| 字幕 | FunASR | 中文语音识别 + 时间轴 |
| 包装 | HyperFrames | 字幕/标题/Logo/B-roll/动画 |
| 合成 | FFmpeg | 最终视频编码导出 |

## 流水线

```
文案 → GPT-SoVITS → audio.wav
                  → MuseTalk → talking.mp4
                 → FunASR → subtitle.json
                → HyperFrames → packaged.mp4
                             → FFmpeg → final.mp4
```

## 快速开始

### 1. 环境检查

```bash
python scripts/check_environment.py
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备素材

- 人物视频放到 `avatar/avatar.mp4`
- 参考音频放到 `voice/references/default.wav`
- Logo 放到 `assets/logo/logo.png`

### 4. 安装核心模型

**GPT-SoVITS:**
```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git voice/models/GPT-SoVITS
cd voice/models/GPT-SoVITS
pip install -r requirements.txt
```

**MuseTalk:**
```bash
git clone https://github.com/TMElyralab/MuseTalk.git app/backend/providers/lipsync/MuseTalk
cd app/backend/providers/lipsync/MuseTalk
pip install -r requirements.txt
```

### 5. 运行流水线

```bash
# 完整运行
python scripts/pipeline.py --title "为什么你的停球总是停不好" --script "为什么你的第一脚触球一直停不好？今天告诉你三个关键点。" --template football_knowledge

# 断点续跑
python scripts/pipeline.py --project 2026-08-07_xxx --resume

# 单独运行各模块
python scripts/generate_voice.py --text "你的文案" --output ./outputs/test/audio.wav
python scripts/generate_lipsync.py --audio ./outputs/test/audio.wav --output ./outputs/test/talking.mp4
python scripts/generate_subtitles.py --audio ./outputs/test/audio.wav --output ./outputs/test/
python scripts/render_video.py --talking ./outputs/test/talking.mp4 --subtitles ./outputs/test/subtitle.json
```

### 6. Web 控制台

```bash
python scripts/web_server.py
# 访问 http://127.0.0.1:8080
```

## 目录结构

```
ai-talking-video/
├── app/
│   ├── frontend/          # Web 前端
│   └── backend/
│       └── providers/     # 核心模块
│           ├── voice/     # GPT-SoVITS 封装
│           ├── lipsync/   # MuseTalk 封装
│           ├── subtitle/  # FunASR 封装
│           └── render/    # HyperFrames 封装
├── avatar/                # 数字人素材
├── voice/                 # 声音模型和配置
├── assets/                # Logo/音乐/B-roll
├── templates/             # 视频模板
├── scripts/               # 运行脚本
├── outputs/               # 输出目录
└── config/                # 配置文件
```

## 输出格式

- 分辨率: 1080×1920 (竖屏)
- 帧率: 30fps
- 编码: H.264 + AAC
- 格式: MP4

## 开发阶段

- [x] Phase 1: MVP 流水线 (文案 → final.mp4)
- [ ] Phase 2: 自动素材系统
- [ ] Phase 3: AI导演一键生成
