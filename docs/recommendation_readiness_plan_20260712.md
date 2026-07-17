# 血流麻将推荐可用性分级开发计划

## 实施状态（2026-07-12）

本计划的软件开发部分已完成：

- 已新增独立的 `RecommendationReadiness`，不再由任意外围 unknown 直接关闭推荐；
- 已实现 `BLOCKED / HAND_ONLY / TABLE_AWARE / ENHANCED` 四档模式；
- 已把手牌、我的副露、定缺和出牌轮次设为核心硬门禁；
- 已将 unknown 按 `CRITICAL / CONTEXTUAL / IGNORABLE` 分类；
- 已实现仅手牌与牌桌增强两次策略计算，并比较首选牌和向听结果；
- 已在实时 UI 和 JSONL 日志中显示模式、核心分、上下文分、稳健性和降级原因；
- 已让回放测试输出推荐模式和质量分；
- 已增加风险-覆盖率计算及阈值选择工具；
- 初始阈值已移入 `configs/app.yaml`；
- 自动化测试覆盖截图对应的“11 张手牌 + 已确认碰牌 + 外围 unknown”场景。

阈值校准命令：

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_recommendation_threshold.py `
  data\gold_replay\recommendation_readiness.jsonl `
  --maximum-risk 0.01 `
  --output reports\recommendation_threshold.json
```

输入 JSONL 每行至少包含：

```json
{"core_score": 95, "core_state_correct": true, "eligible": true}
```

剩余工作属于数据验收：需要在独立人工核对回放集上测量错误放行率和覆盖率，再将正式阈值写回配置。训练集不能替代这一步。

日期：2026-07-12  
阶段：牌桌结构化识别之后的推荐门禁优化  
目标：在保证核心手牌输入可靠的前提下，提高推荐覆盖率，不再因为外围少量未知牌无条件暂停推荐。

---

## 1. 背景与当前问题

当前系统已经具备：

- YOLO 单牌检测；
- 手牌、碰杠、弃牌、事件牌等区域划分；
- 单牌和副露的多帧跟踪；
- 手牌槽位补全；
- 结构化状态校验；
- 异常状态下暂停推荐和保存问题帧。

目前推荐门禁仍然过于保守。`state_validator.py` 中只要存在任意
`unknown_tiles`，就会返回 `UNCERTAIN` 并禁止推荐：

```python
if zones.unknown_tiles:
    return StructuredStateResult(
        RegionState.UNCERTAIN,
        False,
        "unknown_tiles_present",
    )
```

这种规则没有区分未知牌对策略的实际影响：

- 手牌中的未知牌会直接改变向听数，必须阻止推荐；
- 我的碰杠数量不确定会改变合法手牌数量，必须阻止推荐；
- 中央弃牌漏一张只会降低剩余牌估计质量，不应阻止基础推荐；
- 对手碰杠或胡牌展示区域存在少量未知，只影响增强信息；
- 远离核心区域的动画牌或误检，不应与手牌缺失同级处理。

当前策略函数已经支持不提供 `visible_counts`。因此，只要手牌、我的副露数、
定缺和出牌阶段可信，就可以先给出仅基于手牌的基础推荐。外围可见牌用于提高
有效牌数量估计和候选排序质量，而不是基础推荐成立的必要条件。

---

## 2. 核心结论

推荐门禁应从：

```text
牌桌存在任何不确定 -> 暂停全部推荐
```

改为：

```text
核心状态不可靠 -> 暂停推荐
核心状态可靠、外围信息不足 -> 手牌基础推荐
核心状态可靠、外围信息部分可信 -> 普通推荐
核心状态可靠、外围信息充分且结果稳健 -> 增强推荐
```

核心状态包括：

1. 当前轮到我出牌；
2. 定缺花色已经选择；
3. 手牌牌面和数量完整；
4. 我的已确认副露数量正确；
5. 手牌和我的副露不存在牌数、重叠或轨迹冲突。

外围补充信息包括：

- 四家弃牌；
- 对手碰杠；
- 胡牌展示；
- 与核心区域无关的事件牌；
- 其他已见牌统计。

---

## 3. 设计原则

### 3.1 结构状态与推荐可用性分离

`StructuredTableState` 继续如实描述识别状态，可以保持 `UNCERTAIN`、
`PARTIAL` 等状态；新增独立的 `RecommendationReadiness`，专门判断当前是否
足以给出哪一级推荐。

禁止让“牌桌结构不完美”自动等价于“不能提供任何建议”。

### 3.2 核心信息使用硬门禁

任何可能改变手牌构成、向听数、合法出牌集合或副露数量的问题，必须阻止推荐。

### 3.3 外围信息使用软降级

外围信息不足时，降低推荐等级并显示警告，但不取消已经可靠的手牌基础建议。

### 3.4 不使用 YOLO 原始置信度作为唯一阈值

YOLO 的 `0.75` 不是“75%正确概率”。置信度会受到牌尺寸、位置、旋转角度和
遮挡影响。最终阈值必须通过人工核对的回放集进行校准。

### 3.5 优先验证推荐结果是否稳健

同一手牌分别在“忽略外围信息”和“加入可信外围信息”两种场景下计算。如果
推荐牌一致，则说明建议对外围识别误差不敏感，可以提高推荐等级。

---

## 4. 推荐等级

新增以下推荐模式：

```python
class RecommendationMode(str, Enum):
    BLOCKED = "blocked"
    HAND_ONLY = "hand_only"
    TABLE_AWARE = "table_aware"
    ENHANCED = "enhanced"
```

### 4.1 BLOCKED

核心状态不完整或不可信，不调用策略算法。

典型原因：

- 不是我的出牌阶段；
- 未选择定缺；
- 手牌数量不合法；
- 手牌存在未知槽位或不可接受的推断；
- 我的副露组数不确定；
- 核心区域有动画牌；
- 核心牌数超过四张；
- 手牌和我的碰杠区域存在重叠冲突。

### 4.2 HAND_ONLY

核心状态可靠，但外围牌桌信息不足。

策略输入：

```text
完整手牌
我的已确认副露数
定缺花色
visible_counts = {}
```

UI 显示：

```text
基础推荐（仅根据手牌）
```

### 4.3 TABLE_AWARE

核心状态可靠，部分弃牌和对手副露可信。

策略仅使用经过多帧确认的外部可见牌，不使用不稳定 unknown、动画牌和推断牌。

UI 显示：

```text
普通推荐（牌桌信息部分完整）
```

### 4.4 ENHANCED

核心状态可靠，外围信息覆盖率高，并且多种不确定场景下推荐结果一致。

UI 显示：

```text
增强推荐（牌桌信息充分）
```

---

## 5. 数据结构

新增模块：

```text
src/xueliu_ai/table/recommendation_readiness.py
```

建议数据结构：

```python
@dataclass(frozen=True)
class RecommendationReadiness:
    mode: RecommendationMode
    allow_recommend: bool
    core_score: float
    context_score: float
    robust: bool
    hard_block_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
```

未知牌按影响范围分类：

```python
class UnknownImpact(str, Enum):
    CRITICAL = "critical"
    CONTEXTUAL = "contextual"
    IGNORABLE = "ignorable"
```

分类含义：

- `CRITICAL`：位于手牌、摸入槽位或我的碰杠结构中；
- `CONTEXTUAL`：位于弃牌区、对手碰杠区或胡牌展示锚点附近；
- `IGNORABLE`：短时动画、UI误检、牌墙或牌桌结构外检测。

---

## 6. 核心状态硬门禁

满足以下全部条件才允许至少进入 `HAND_ONLY`：

### 6.1 出牌阶段正确

```text
phase == MY_TURN
```

摸牌动画、等待其他玩家、结算和加载状态仍然禁止推荐。

### 6.2 定缺已选择

```text
missing_suit is not None
```

### 6.3 手牌数量正确

```text
expected_hand_count = 14 - 3 * confirmed_open_melds
len(hand) == expected_hand_count
```

不能使用对手副露数或 suspected meld 修改该数量。

### 6.4 手牌槽位完整且稳定

默认要求：

- 手牌槽位连续稳定至少 3 帧；
- 没有未知类别槽位；
- 没有中间空槽；
- 同一物理牌没有重复轨迹；
- 所有牌属于合法麻将牌类别。

允许的安全推断：

- 某一槽位仅短时漏检；
- 该槽位此前至少连续 5 帧类别和位置稳定；
- 推断不超过 1 张；
- 下一帧仍需继续验证；
- 推断牌不会伪装成实际观测牌进入已见牌统计。

### 6.5 我的副露数量稳定

- 只使用 `PONG` 和 `KONG` 计算 `confirmed_open_melds`；
- `SUSPECTED_PONG` 和 `SUSPECTED_KONG` 如果会改变期望手牌数量，必须阻止推荐；
- 我的副露组需要连续稳定至少 3 帧；
- 我的碰杠区中存在未解决孤立牌时禁止推荐。

### 6.6 无核心结构矛盾

以下任一情况直接阻止：

- 手牌加我的实际副露中，同类牌实际观测超过 4 张；
- 一个 track 同时属于手牌和碰杠；
- 手牌 ROI 与我的碰杠 ROI 对同一检测框发生冲突；
- 核心区域存在活动动画牌；
- `meld_groups` 与最终 `zone_tiles` 不一致。

---

## 7. 核心可靠度评分

除硬门禁外，计算 0 到 100 的 `core_score`：

| 项目 | 分值 |
|---|---:|
| 手牌数量与槽位完整 | 35 |
| 手牌多帧标签和位置稳定 | 25 |
| 我的副露组数和结构稳定 | 20 |
| 出牌阶段及定缺状态正确 | 10 |
| 无牌数、轨迹和区域冲突 | 10 |

初始门槛：

```text
core_score >= 90
```

同时必须通过全部硬门禁。不能通过其他项目的分数抵消手牌数量错误。

该 90 分只是第一版工程阈值，最终应由回放集的风险-覆盖率曲线确定。

---

## 8. 外围信息质量评分

计算独立的 `context_score`，它只决定推荐等级，不决定基础推荐是否存在。

建议权重：

| 项目 | 权重 |
|---|---:|
| 中央及四家弃牌稳定覆盖率 | 40% |
| 对手碰杠组稳定覆盖率 | 30% |
| 外部轨迹连续性与重复框控制 | 20% |
| 胡牌展示和事件牌已正确隔离 | 10% |

初始分级：

```text
context_score >= 80 -> ENHANCED 候选
50 <= context_score < 80 -> TABLE_AWARE
context_score < 50 -> HAND_ONLY
```

注意：

- `unknown_tiles` 数量不能直接作为硬阈值；
- unknown 必须结合位置、持续时间和是否进入核心区域判断；
- 外围 unknown 再多也不能污染 `visible_counts`；
- 场景明显处于动画、结算或非牌局状态时，仍由阶段状态阻止推荐。

---

## 9. 双场景稳健性校验

核心状态通过后，至少运行两次策略：

### 场景 A：仅手牌

```python
advice_hand_only = advise_discard(
    hand,
    missing_suit,
    visible_counts={},
    open_melds=confirmed_open_melds,
)
```

### 场景 B：可信牌桌信息

```python
advice_table_aware = advise_discard(
    hand,
    missing_suit,
    visible_counts=trusted_observed_visible_counts,
    open_melds=confirmed_open_melds,
)
```

可选场景 C：把连续稳定的外围 unknown 按其检测类别加入，用于敏感性分析，但
不能直接作为正式已见牌统计。

稳健性规则：

```text
A 与 B 推荐相同 -> robust = True
A、B、C 推荐均相同 -> 可进入 ENHANCED
A 与 B 不同 -> 保留推荐，但降级并同时展示两个候选
```

当结果不一致时，UI 建议显示：

```text
基础推荐：7万
结合牌桌：8万
提示：外围可见牌可能影响候选排序
```

不要因为外围信息导致两个候选不同就完全隐藏手牌基础结果。

---

## 10. 推荐执行链路

目标链路：

```text
YOLO检测
  -> 单牌跟踪
  -> 最终区域归属
  -> 碰杠与手牌槽位融合
  -> StructuredTableState
  -> 结构一致性校验
  -> RecommendationReadiness
       -> 核心状态失败：BLOCKED
       -> 核心通过：运行手牌基础策略
       -> 外围可信：运行牌桌增强策略
       -> 比较结果稳健性
  -> UI显示等级、建议和警告
```

`StructuredStateResult.allow_recommend` 不再单独决定最终结果。最终门禁改为：

```python
allow_recommend = readiness.allow_recommend
```

结构状态仍作为 `readiness` 的输入和诊断来源。

---

## 11. 文件修改计划

### 11.1 新增

```text
src/xueliu_ai/table/recommendation_readiness.py
tests/test_recommendation_readiness.py
```

### 11.2 修改

```text
src/xueliu_ai/table/state_validator.py
src/xueliu_ai/table/game_phase.py
src/xueliu_ai/ui/realtime_app.py
src/xueliu_ai/evaluation/video_replay.py
src/xueliu_ai/evaluation/replay_test.py
```

### 11.3 原则上不修改

```text
麻将向听算法
有效牌算法
现有弃牌评分公式
YOLO权重和训练配置
```

本阶段只重构“什么时候调用策略、使用多少上下文、如何表达可信度”。

---

## 12. UI 调整

实时结果区增加：

```text
推荐模式：基础 / 普通 / 增强 / 暂停
核心可靠度：0-100
牌桌信息完整度：0-100
稳健性：一致 / 对外围信息敏感
警告：外围未知2张，已忽略，不影响基础推荐
```

推荐展示建议：

- 绿色：增强推荐；
- 蓝色：普通推荐；
- 黄色：手牌基础推荐；
- 红色：核心状态异常，暂停推荐。

禁止继续只显示笼统的：

```text
unknown_tiles_present
```

应显示具体影响：

```text
外围未知2张，已降级为手牌基础推荐
```

或：

```text
手牌区域存在1张未知牌，无法计算可靠向听，暂停推荐
```

---

## 13. 测试计划

### 13.1 核心门禁测试

1. 手牌完整、外围 unknown 两张，允许 `HAND_ONLY`；
2. 手牌缺一张，必须 `BLOCKED`；
3. 摸入牌槽位未知，必须 `BLOCKED`；
4. 我的副露组数未确认，必须 `BLOCKED`；
5. 对手碰杠存在 suspected group，不阻止 `HAND_ONLY`；
6. 核心区域存在动画牌，必须 `BLOCKED`；
7. 手牌和我的副露同牌实际数量超过四张，必须 `BLOCKED`；
8. 未选择定缺，必须 `BLOCKED`；
9. 等待其他玩家时，必须 `BLOCKED`；
10. 允许的单张短时手牌推断稳定后进入安全模式。

### 13.2 外围降级测试

1. 外围信息充分时进入 `TABLE_AWARE` 或 `ENHANCED`；
2. 弃牌漏检时降级但继续推荐；
3. 胡牌展示归属不确定时降级但继续推荐；
4. unknown 不进入正式 `visible_counts`；
5. 外围轨迹重复不会让同类牌数量超过四张；
6. 大量外围不确定但核心完整时仍能给出 `HAND_ONLY`；
7. 非牌局或结算画面不会因为手牌历史残留而给出推荐。

### 13.3 稳健性测试

1. 手牌场景和牌桌场景推荐相同，标记 `robust=True`；
2. 两种场景推荐不同，降级并保留两个候选；
3. 未知牌加入与否不改变第一候选时，允许增强推荐；
4. 推荐候选列表顺序抖动但第一候选不变，不重置稳定状态；
5. 推荐第一候选连续变化时，不显示增强推荐。

### 13.4 回归测试

所有当前测试必须保持通过，特别包括：

- 手牌中间漏检恢复；
- 真实出牌后历史牌过期；
- 2筒碰组不混入手牌；
- 7万碰组多帧稳定；
- 推断牌不进入实际已见牌统计；
- 动画牌不进入策略输入；
- 最终区域和预览区域一致。

---

## 14. 回放校准与阈值确定

### 14.1 校准集

从现有人工核对视频和问题帧建立独立推荐门禁校准集，不能使用训练集代替。

至少覆盖：

```text
无副露13/14张
一个碰后的10/11张
一个杠后的10/11张
多个副露
右侧独立摸入牌
手牌单帧漏检
手牌真实出牌
外围弃牌漏检
对手碰杠漏检
胡牌展示
飞牌动画
加载和结算画面
```

### 14.2 标注内容

每个关键帧标注：

```text
是否轮到我出牌
真实手牌
我的副露组数
定缺状态
是否应允许基础推荐
可信外围已见牌
正确的推荐模式
```

### 14.3 指标

核心指标不再是单纯的 YOLO mAP，而是：

| 指标 | 初始目标 |
|---|---:|
| 被允许推荐帧的核心手牌正确率 | >=99% |
| 错误放行推荐率 | <=1% |
| 我的出牌阶段基础推荐覆盖率 | >=85% |
| 核心完整但因外围 unknown 被错误阻止 | 0 |
| 动画覆盖核心区域时错误放行 | 0 |
| HAND_ONLY 与 TABLE_AWARE 推荐一致率 | 记录基线后持续提高 |

### 14.4 阈值选择

遍历不同 `core_score` 阈值，绘制：

```text
推荐覆盖率 coverage
对
错误推荐风险 risk
```

选择满足核心正确率至少 99% 时覆盖率最高的阈值。初始使用 90 分，待校准后
写入配置文件，而不是长期硬编码。

建议配置：

```yaml
recommendation_readiness:
  minimum_core_score: 90
  table_aware_context_score: 50
  enhanced_context_score: 80
  minimum_core_stable_frames: 3
  inferred_hand_history_frames: 5
  max_safe_inferred_hand_tiles: 1
```

---

## 15. 分阶段实施

### P0：硬门禁与外围软降级

- 新增 `RecommendationReadiness`；
- 将 unknown 按 `CRITICAL/CONTEXTUAL/IGNORABLE` 分类；
- 移除“任意 unknown 均禁止推荐”的逻辑；
- 核心完整时允许 `HAND_ONLY`；
- 增加核心门禁测试。

验收：截图中的“手牌11张、一个已确认碰组、外围unknown=2”可以给出基础推荐。

### P1：推荐等级和 UI

- 增加四档推荐模式；
- 计算 `core_score` 和 `context_score`；
- UI 显示具体降级原因；
- 日志记录推荐模式和分数。

### P2：双场景稳健性

- 同时计算手牌基础建议和牌桌增强建议；
- 比较第一候选、向听和候选分差；
- 不一致时降级并展示两个候选；
- 增加稳健性回归测试。

### P3：视频回放校准

- 建立人工核对校准集；
- 输出风险-覆盖率曲线；
- 根据目标错误率确定正式阈值；
- 将阈值写入配置。

### P4：灰度验证

- 默认仍保存被降级或阻止的问题帧；
- 连续回放完整真实牌局；
- 对比旧门禁和新门禁的推荐覆盖率；
- 人工抽查 HAND_ONLY 帧；
- 达标后替换旧门禁。

---

## 16. 验收标准

开发完成必须同时满足：

```text
1. 核心手牌完整时，外围少量 unknown 不再无条件暂停推荐。
2. 手牌不完整、我的副露数不确定时仍然严格阻止推荐。
3. HAND_ONLY 模式不读取不可信外围 visible_counts。
4. unknown、动画牌和推断牌不会污染实际已见牌统计。
5. UI 明确显示推荐等级和降级原因。
6. 手牌基础与牌桌增强结果不一致时能明确提示。
7. 推荐门槛通过回放集校准，而不是依赖单一 YOLO 置信度。
8. 错误放行推荐率不高于1%。
9. 所有现有和新增自动化测试通过。
```

---

## 17. 本阶段不做的工作

- 不重新训练 YOLO；
- 不通过降低置信度解决区域问题；
- 不修改向听算法；
- 不大幅修改弃牌评分公式；
- 不把推断牌伪装成真实检测；
- 不因为提高覆盖率而取消核心安全门禁；
- 不使用训练数据直接评估推荐门槛。

---

## 18. 参考方法

- Selective Classification for Deep Neural Networks：通过期望风险与覆盖率权衡决定何时拒绝预测。  
  https://arxiv.org/abs/1705.08500
- On Calibration of Modern Neural Networks：神经网络原始置信度通常不能直接解释为正确概率。  
  https://proceedings.mlr.press/v70/guo17a.html
- Multivariate Confidence Calibration for Object Detection：目标检测置信度会受到位置和框尺寸影响。  
  https://openaccess.thecvf.com/content_CVPRW_2020/html/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.html
- Decision Theoretic Foundations for Conformal Prediction：在不确定输入下使用风险敏感、最坏场景稳健的决策方法。  
  https://proceedings.mlr.press/v267/kiyani25a.html
