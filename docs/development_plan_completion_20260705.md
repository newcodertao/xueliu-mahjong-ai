# 下一步开发计划完成记录

日期：2026-07-05

参考文档：`docs/血流麻将AI辅助项目下一步开发计划.md`

## 已完成

1. 清理敏感文件和缓存
   - 已删除本地 `.env`、`IP账号密码.txt`。
   - 已清理 `__pycache__`、`.pytest_cache`、临时 replay report。
   - `.gitignore` 增加账号、密码、token、secret 相关规则。

2. 统一运行主模型
   - 默认模型统一为 `models/yolo/xueliu_final325_clean_v1_0703.pt`。
   - `configs/model.yaml`、`README.md`、CLI 默认值均指向 final325。
   - `models/yolo/training_rounds.yaml` 重写为 clean-start 主线。

3. 黄金回放测试
   - 新增 `data/gold_replay/gold_cases.json` 和说明。
   - 新增 `python -m xueliu_ai replay-test`。
   - 支持输出阶段、推荐开关、手牌数量、碰杠数量、诊断有效性指标。
   - 失败样本会输出到 `reports/replay_failures/YYYYMMDD_HHMMSS/`。

4. 牌局阶段和推荐门控
   - 新增 `src/xueliu_ai/table/game_phase.py`。
   - 支持加载中、发牌中、等待定缺、等待摸牌、轮到我、结算/非牌局等阶段。
   - 只有轮到我、定缺已选、识别稳定、区域诊断通过时才允许推荐。

5. 我的区域分析
   - 新增 `src/xueliu_ai/table/my_area.py`。
   - 输出手牌、摸牌、碰杠区、碰杠组数、合法手牌数量、异常候选。

6. 策略能力补充
   - 新增定缺建议。
   - 新增换三张建议。
   - 增强碰牌、杠牌建议。
   - 增强基础番型摘要。

7. UI 改进
   - 实时 UI 增加牌局阶段。
   - 实时 UI 增加推荐开关和暂停原因。
   - 推荐逻辑接入阶段门控，不再在发牌/区域异常/未定缺/未稳定时强行推荐。

8. 打包
   - 新增 `scripts/package_runtime.py`。
   - 已重新生成 `F:\xueliu-mahjong-ai_runtime_package_20260705.zip`。
   - 包内不包含 `data/`、`datasets/`、`runs/`、`reports/`、`.git`、`.venv`、`.env`、账号密码文件。

## 未执行的部分

硬样本新模型训练没有执行，因为当前没有新的、已人工确认的硬样本标签。

下一轮应先准备：

- 动态手牌/碰杠边界样本
- 摸牌分离样本
- 左右/对家碰杠与弃牌归属样本
- 动画遮挡和低置信样本

这些样本需要人工确认后再训练，训练完成后必须通过 `replay-test` 不退化，才能切换主模型。

## 验证

- `python -m pytest`：44 passed
- `python -m xueliu_ai replay-test --gold data/gold_replay/gold_cases.json --no-images`：命令通过，当前黄金集为空
- `python scripts/package_runtime.py`：打包通过，自检无禁入文件

