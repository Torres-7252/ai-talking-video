# 初始化

# 创建 avatar 占位说明
echo "请将人物视频 (avatar.mp4) 放到此目录" > avatar/README.txt
echo "" >> avatar/README.txt
echo "要求:" >> avatar/README.txt
echo "  - 正面人物，胸部以上" >> avatar/README.txt
echo "  - 自然眨眼、轻微头部和身体动作" >> avatar/README.txt
echo "  - 背景稳定，嘴部无遮挡" >> avatar/README.txt

# 创建 voice references 占位说明
echo "请将参考音频 (default.wav) 放到此目录" > voice/references/README.txt
echo "" >> voice/references/README.txt
echo "参考音频用于 GPT-SoVITS 克隆音色" >> voice/references/README.txt
echo "建议 5-10 秒，清晰中文语音" >> voice/references/README.txt
