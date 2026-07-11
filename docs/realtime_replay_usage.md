# 实时识别与视频回放

## 实时识别

启动：

```powershell
F:\xueliu-mahjong-ai\血流麻将实时辅助.exe
```

建议保持：

- 置信度：0.75
- IOU：0.45
- 自动识别手牌/碰杠/弃牌区域：开启
- 显示实时预览窗口：开启

预览窗口会显示：

- `hand`：当前暗手牌数量
- `my_meld`：我的碰杠区数量
- `left_meld/top_meld/right_meld`：其他三家碰杠区数量
- `center`：中间弃牌数量
- `discard areas`：四家弃牌区归属数量

如果出现手牌数量异常、牌数超过 4 张、碰杠区数量可疑，UI 会显示“区域异常，暂停推荐”。

## 视频回放测试

用视频批量回测识别、分区、异常和推荐：

```powershell
cd F:\xueliu-mahjong-ai
.\.venv\Scripts\python.exe -m xueliu_ai video-replay `
  --video "G:\video\your_video.mp4" `
  --output "data\replays\your_video_test" `
  --every-seconds 1 `
  --max-frames 120
```

输出：

- `data\replays\your_video_test\replay.jsonl`：每帧识别、分区、诊断、推荐
- `data\replays\your_video_test\summary.json`：汇总
- `data\replays\your_video_test\overlays\`：叠框图片

如果只要 JSON，不保存图片：

```powershell
.\.venv\Scripts\python.exe -m xueliu_ai video-replay --video "G:\video\your_video.mp4" --no-images
```
