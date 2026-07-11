# xueliu-mahjong-ai

四川血流成河麻将本地视觉识别与出牌教学系统。

当前版本是可运行的第一阶段 MVP：

- 屏幕截图采集
- ROI 配置与裁剪
- YOLO 数据集目录与 `data.yaml` 生成
- YOLO 手牌 ROI 检测封装
- 检测结果规则校验与连续帧稳定确认
- 27 张数牌编码、向听数、有效进张
- 基础出牌推荐与中文解释
- JSONL 对局日志
- 样本清单、检测导出、基准评测和复盘报告
- CLI 与 pytest 测试

## 安装

```powershell
cd F:\xueliu-mahjong-ai
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[dev]
```

## 常用命令

```powershell
python -m xueliu_ai build-dataset
python -m xueliu_ai collect --interval 1 --limit 10
python -m xueliu_ai manifest
python -m xueliu_ai detect --model models/yolo/xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt --source data/raw/my_hand
python -m xueliu_ai benchmark --model models/yolo/xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt
python -m xueliu_ai advise --hand 1W,2W,3W,4W,5W,6W,7T,8T,9T,2B,3B,4B,9B,9B --missing-suit W
python -m xueliu_ai realtime --model models/yolo/xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt --missing-suit W
python -m xueliu_ai debug-viewer --image data/raw/fullscreen/example.png
python -m xueliu_ai report --log data/games/session.jsonl
pytest
```

## 你需要准备的东西

后续训练 YOLO 需要真实样本：

- 先运行 `python -m xueliu_ai roi-calibrate --name my_hand` 标定手牌区域。
- 再运行 `python -m xueliu_ai collect --interval 0.5 --limit 200` 采集手牌截图。
- 把采集到的 `data/raw/my_hand` 图片用 CVAT、Roboflow 或 Label Studio 标注为 YOLO 格式。
- 当前运行主模型为 `models/yolo/xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt`。

## 边界

本项目只基于本地截图做视觉识别、教学提示和复盘。不自动点击，不读取进程内存，不抓包，不修改客户端，不绕过反作弊。实时模式建议用于单机练习、自建教学局、离线录像或非竞技环境。
