# 本地 AI 口播视频生成器

在 Windows 和 NVIDIA GPU 上本地生成中文 AI 口播视频。流水线使用 GPT-SoVITS v3 克隆参考音色、MuseTalk 1.5 生成嘴型、FunASR 生成时间轴，并由 FFmpeg 输出带字幕的横屏 MP4；不调用付费云端语音或视频 API。

## 输出规格

- 画面：1920x1080，25 fps，保留完整人物和背景
- 视频：H.264，`yuv420p`，limited range
- 音频：AAC
- 文件：`outputs/<项目名>/final.mp4`

## 首次准备

在 PowerShell 中进入项目目录：

```powershell
cd D:\workspaces\ai-talking-video
pip install -r requirements.txt
```

下载 MuseTalk 1.5 及其 VAE、Whisper、人脸检测依赖。下载量为数 GB，脚本支持断点续传：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_musetalk_runtime.ps1
powershell -ExecutionPolicy Bypass -File scripts\download_musetalk_models.ps1
```

准备人物图片和参考声音。以下命令不会修改桌面上的原文件：

```powershell
python scripts\prepare_assets.py `
  --image "C:\Users\31078\Desktop\60b37dcd61ff34e82bace8beb9a5b679.jpg" `
  --voice "C:\Users\31078\Desktop\标准录音 1.mp3"
```

项目使用：

- `avatar/avatar.jpg`：人物和完整背景
- `voice/references/default.wav`：3 至 10 秒清晰参考声音
- `voice/references/default.json`：参考录音对应的原文

检查 GPU、FFmpeg、Python 库、素材和每个模型文件：

```powershell
python scripts\check_environment.py
```

只有全部项目显示 `[OK]` 后才开始生成。首次模型下载时长取决于网络；首次推理还会建立模型和人物坐标缓存。

## 启动网页

运行：

```powershell
.\启动控制台.ps1
```

控制台地址为 [http://127.0.0.1:8080](http://127.0.0.1:8080)。人物图片、参考音频、参考原文和口播文案均可在页面中管理。

## 命令行生成

短片验收命令：

```powershell
python scripts\pipeline.py `
  --project acceptance `
  --title "本地AI口播测试" `
  --script "大家好，这是本地AI口播视频测试。今天我们一起练好第一脚触球。" `
  --template talking_head
```

失败后可从已有项目续跑，无需再次粘贴文案：

```powershell
python scripts\pipeline.py --project acceptance --resume
```

流水线只跳过通过媒体探测的完整产物；空文件或损坏文件会自动重新生成。每一步的状态、错误、日志路径和媒体摘要保存在 `outputs/<项目名>/metadata.json`。

## 单模块命令

```powershell
python scripts\generate_voice.py --text "大家好，这是本地口播测试。" --output outputs\acceptance\audio.wav
python scripts\generate_lipsync.py --avatar avatar\avatar.jpg --audio outputs\acceptance\audio.wav --output outputs\acceptance\talking.mp4
python scripts\generate_subtitles.py --audio outputs\acceptance\audio.wav --output outputs\acceptance\subtitle.json
python scripts\render_video.py --video outputs\acceptance\talking.mp4 --subtitles outputs\acceptance\subtitle.json --output outputs\acceptance\final.mp4
```

## 常见问题

- `MuseTalk 1.5 model files are incomplete`：重新运行模型下载脚本，直到环境检查通过。
- `CUDA out of memory`：关闭占用 GPU 的程序后重试；项目默认使用 FP16 和 batch size 1，并在进入 MuseTalk 前卸载 GPT-SoVITS。
- `ffmpeg` 或 `ffprobe` 未找到：安装 FFmpeg，并将其 `bin` 目录加入 `PATH`。
- FunASR 首次运行较慢：主识别模型和 VAD 会下载到用户缓存，后续运行直接复用。
- 生成中断：保留项目目录，使用 `--resume` 继续。

## 验证

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app scripts tests
python scripts\check_environment.py
```

请仅使用你有权使用的人物图片和声音录音。
# 自然动作增强

项目支持 `LivePortrait -> MuseTalk 1.5` 双阶段模式。LivePortrait 负责克制的眨眼、眼神、表情和头部微动作，MuseTalk 继续根据克隆语音生成中文口型。默认动作模式为 `natural`，原有仅口型路径可通过 `--motion-mode off` 使用。

自然动作模式会保存 MuseTalk 的逐帧人脸框，并在嘴部区域软融合 35% 的动作底片细节。这样可以保留语音口型，同时减少持续张嘴、嘴内暗块和唇纹模糊；遮罩会跟随人脸移动，不依赖固定画面坐标。

LivePortrait 使用独立 Python 3.10 环境，不会改动主项目的 PyTorch 依赖：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_liveportrait_runtime.ps1
python scripts\check_environment.py
```

安装脚本固定使用 LivePortrait 提交 `9b294b3d0536135442ea73cb01e6cb3ca7029dd3`，下载官方模型，并从官方示例驱动生成本地 `steady` 动作模板。虚拟环境、第三方源码、权重、驱动视频和动作模板均不会提交到 Git。

命令行示例：

```powershell
python scripts\pipeline.py --project realism-natural --title "自然口播" --script "大家好，今天我们来聊一个实用的话题。" --motion-mode natural --motion-style steady --motion-intensity 0.35
python scripts\pipeline.py --project realism-fast --title "快速口播" --script "大家好，今天我们来聊一个实用的话题。" --motion-mode off
```

如安装脚本提示缺少 Python 3.10，先执行：

```powershell
winget install Python.Python.3.10
```
