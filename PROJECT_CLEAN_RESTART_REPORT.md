# 雪流麻将 AI 项目清理、数据集审计与重训计划

生成时间：2026-06-30

## 当前结论

当前训练链路建议停止继续沿用。原因不是模型完全不可用，而是训练过程已经混入多类未经充分人工确认的数据，尤其是自动预标数据。`2T -> 4T` 这类高置信错标如果进入训练集，会让后续模型更自信地犯同类错误。

新的主线应该从 `clean_v1` 重新开始：

- 外部带标注数据只作为候选数据，不直接训练。
- 视频截图全部人工标注后才允许进入训练集。
- 自动标注只能作为参考图或候选框，不再作为训练真值。
- 固定一套人工标注 `gold_test_set`，永远不参与训练。

## 已执行的安全清理

本次只做非破坏性清理，未删除真实数据、模型权重、视频或人工标注。

- 删除缓存目录：`.pytest_cache/`、`.ruff_cache/`
- 删除空的失败下载目录：
  - `external_datasets/roboflow`
  - `external_datasets/roboflow_mahjong_tile_selected_images_2_v1`
  - `external_datasets/roboflow_project_xv49e_mahjong_v1`
- 新建 clean 重训目录：
  - `data/clean_start/gold_test_set/images_to_label`
  - `data/clean_start/train_manual_set/images_to_label`
  - `datasets/clean_v1`
  - `models/yolo/clean_v1`

## 模型链路状态

| 模型 | 状态 | 说明 |
| --- | --- | --- |
| `xueliu_tiles_v1.pt` | legacy | 早期 Roboflow 训练模型，不作为 clean 主线起点 |
| `xueliu_tiles_phone_landscape_human_round1.pt` | legacy/reference | 少量人工样本训练，可参考但不继续沿用 |
| `xueliu_tiles_rotated_plus_human_round1_640.pt` | legacy/reference | 多角度增强旧模型，可参考 |
| `xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7.pt` | legacy | 混入外部数据 |
| `xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7_merged_dense_v1.pt` | legacy | 混入外部 dense 数据 |
| `xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7_merged_dense_chinese_v1.pt` | legacy/reference | 用户反馈一度提升，但仍基于外部未人工审核数据 |
| `xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7_merged_dense_chinese_douyin_reviewed_v1.pt` | contaminated / stop | 基于自动预标后少量人工修正样本继续训练，存在脏数据风险 |

建议：旧模型全部冻结。后续只作为对照，不再作为 clean 主线继续训练。

## 已转换/派生训练数据集

| 数据集 | 图片 | 框数 | 大小 | 风险等级 | 处理建议 |
| --- | ---: | ---: | ---: | --- | --- |
| `phone_landscape_human_round1` | 24 | 357 | 6.4MB | 中 | 样本少，可人工复查后少量并入 clean |
| `xueliu_tiles_douyin_reviewed_v1` | 66 | 2772 | 12.1MB | 高 | 来自自动预标后人工局部修正，不直接并入 clean |
| `xueliu_tiles_video_pseudo_v1` | 399 | 4194 | 132.5MB | 高 | 伪标签数据，禁止进入 clean |
| `xueliu_tiles_roboflow_v1` | 7103 | 84772 | 1.6GB | 高 | 外部未人工审核，不直接训练 |
| `xueliu_tiles_roboflow_yolo_mahjong_v7` | 3233 | 9210 | 243.2MB | 高 | 外部未人工审核，不直接训练 |
| `xueliu_tiles_roboflow_mahjong_tiles_merged_v1` | 3441 | 45034 | 186.2MB | 高 | 外部未人工审核，不直接训练 |
| `xueliu_tiles_chinese_detection_v4_rotated` | 16088 | 213812 | 1.1GB | 高 | 外部未人工审核且旋转增强，不直接训练 |
| `xueliu_tiles_merged_dense_v1_rotated` | 13764 | 180136 | 1.1GB | 高 | 外部数据派生，不直接训练 |
| `xueliu_tiles_rotated_plus_human_round1` | 28508 | 340516 | 9.7GB | 高 | 混合旧数据和增强数据，不直接训练 |
| `xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7` | 41440 | 377356 | 11.1GB | 高 | 旧主线大混合数据，不直接训练 |

## 外部原始数据源

| 外部源 | 图片 | 框数 | 类别数 | 来源 | 建议 |
| --- | ---: | ---: | ---: | --- | --- |
| `roboflow_mahjong` | 7650 | 116169 | 42 | Jon Chan Mahjong v83 | 只抽样人工审核，不直接训练 |
| `roboflow_chinese_mahjong_detection_v4` | 4456 | 76890 | 42 | Chinese Mahjong Detection v4 | 只抽样人工审核，不直接训练 |
| `roboflow_mahjong_tiles_merged_v1` | 3942 | 65661 | 42 | Mahjong Tiles Merged v1 | 只抽样人工审核，不直接训练 |
| `roboflow_yolo_mahjong_v7` | 4012 | 11405 | 38 | YOLO Mahjong v7 | 只抽样人工审核，不直接训练 |
| `roboflow_riichi_mahjong_tiles_v1` | 256 | 6644 | 34 | 日麻牌数据 | 和目标平台差异大，不建议主用 |
| `roboflow_majsoul_v1` | 137 | 1782 | 37 | 雀魂数据 | 平台差异大，不建议主用 |
| `roboflow_mahjong_tiles_oc9zz_v11` | 55 | 1798 | 34 | 小数据集 | 价值有限 |
| `roboflow_mahjong_yolo_v1` | 218 | 3647 | 86 | 类别体系混乱 | 不建议使用 |
| `mahjong-dataset` | 2343 | 无 YOLO 标签 | 未知 | 图片集 | 可人工挑图，不直接训练 |

## 当前标注工作区

| 目录 | 图片 | JSON | 说明 | 建议 |
| --- | ---: | ---: | --- | --- |
| `douyin_20260630_raw500` | 421 | 0 | 500 抽帧后人工初筛剩余 | 可作为 clean 候选原图 |
| `douyin_20260630_prelabeled_chinese_v1` | 842 | 421 | 自动预标副本，含预览图 | 不作为训练真值 |
| `douyin_20260630_relabel_after_reviewed_v1` | 710 | 355 | 低阈值 0.40 重标 | 不作为训练真值 |
| `douyin_20260630_relabel_after_reviewed_v1_conf065` | 710 | 355 | 阈值 0.65 重标 | 只作参考 |
| `douyin_20260630_relabel_after_reviewed_v1_conf085` | 710 | 355 | 阈值 0.85 重标 | 只作参考 |
| `douyin_20260630_remaining_unreviewed_for_relabel` | 355 | 0 | 剩余未处理原图 | 可重新进入人工标注 |
| `phone_landscape_round2_split_20260630` | 320 | 160 | 包含 human_review 与 holdout | holdout 可参考，需人工确认 |
| `dense_tiles_hardcases_20260630` | 146 | 0 | 密集牌难例候选 | 适合 clean 人工标注 |

注意：统计中出现“图片数翻倍”的目录，通常是因为包含 `_preview_boxes` 预览图。真实训练时不能把预览图当原图使用。

## clean_v1 训练原则

1. 训练数据只接受人工确认标签。
2. 自动预标文件、伪标签、模型输出不能直接进入训练集。
3. 外部 Roboflow 数据不能直接进入训练集，必须抽样审核后再决定。
4. `gold_test_set` 必须人工标注，且永远不参与训练。
5. 每轮训练必须保留：
   - 数据集版本
   - 模型权重
   - 训练参数
   - 测试集评估结果
   - 可回退上一轮模型
6. 易混淆类别必须重点复核：
   - `2T / 4T`
   - `5T / 6T / 8T / 9T`
   - 横放牌、侧向碰牌、密集叠牌

## clean_v1 数据准备计划

### 阶段 1：固定测试集

目标：建立一套永久不参与训练的标准考卷。

- 从真实视频/直播截图中选择 `100` 张。
- 必须覆盖：
  - 自己手牌
  - 左/右/上家副露
  - 中间出牌堆
  - 横放/旋转牌
  - UI 弹窗干扰
  - 密集贴牌
- 人工完整标注所有可识别的 27 类数牌。
- 输出目录：
  - `data/clean_start/gold_test_set/images_to_label`
  - 后续导出到 `datasets/clean_v1_gold_test`

### 阶段 2：第一版人工训练集

目标：先做小而干净的训练集。

- 从真实视频/直播截图中选择 `300-500` 张。
- 只用人工手动标注，不使用自动预标作为真值。
- 每张图只标肉眼能确认的牌；看不清的不标。
- 输出目录：
  - `data/clean_start/train_manual_set/images_to_label`
  - 后续导出到 `datasets/clean_v1`

### 阶段 3：训练 clean_v1

推荐参数：

- 起始模型：`yolo11n.pt` 或干净 COCO 预训练模型，不从旧麻将模型继续训练。
- `imgsz=960`
- `batch=12` 或 `batch=16`
- `workers=4`
- `epochs=80`
- `patience=15`
- 不使用旧伪标签数据。

### 阶段 4：评估

只用 `gold_test_set` 评估。

必须单独记录：

- 总体 mAP50 / mAP50-95
- `2T` 与 `4T` 混淆情况
- 副露区漏检率
- UI 误检数量
- 密集牌漏检情况

### 阶段 5：第二轮补强

只把人工确认过的新难例加入训练：

- `2T/4T` 混淆样本
- 横放筒子牌
- 左右侧碰牌
- 多张紧贴的出牌堆
- UI 弹窗负样本

## 建议下一步

1. 从现有 `douyin_20260630_raw500/images_to_screen` 或新视频中挑 `100` 张做 `gold_test_set`。
2. 再挑 `300-500` 张做 `train_manual_set`。
3. 使用 X-AnyLabeling 纯人工标注，不预标。
4. 标注完成后再构建 `datasets/clean_v1` 并从干净预训练模型开始训练。

一句话原则：宁可慢一点，也要让 clean_v1 的每一个训练标签都可信。
