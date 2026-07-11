# 血流麻将 AI 辅助项目下一步开发计划

日期：2026-07-05  
项目定位：本地运行的血流麻将实时识别与出牌教学辅助系统  
当前阶段：模型基本可用，重点转向实时区域归属、牌局状态机、回放测试和策略稳定性

---

## 0. 当前项目结论

当前项目已经完成了最小可用原型：

- 可以从屏幕或投屏窗口实时截图。
- 可以使用 YOLO 识别麻将牌。
- 已有基础区域划分能力。
- 已有实时 UI。
- 已有稳定帧过滤。
- 已有基础出牌建议。
- 已有视频回放工具。
- 当前主模型为：

```text
models/yolo/xueliu_final325_clean_v1_0703.pt
```

当前最重要的问题不是继续盲目训练模型，而是：

```text
1. 清理交付包
2. 统一模型路径和配置
3. 建立黄金视频回放测试集
4. 增强牌局阶段识别
5. 增强实时区域归属稳定性
6. 再补充 hard case 数据训练新模型
7. 最后升级策略层
```

---

## 1. 总体开发原则

### 1.1 不再闭门造车

每次修改区域逻辑、模型、策略，都必须通过固定回放测试验证。

### 1.2 不使用未经人工确认的数据训练

模型可以预标注，但所有自动标注结果必须人工确认后才能进入训练集。

禁止：

```text
模型自动标注 -> 不人工检查 -> 直接继续训练
```

允许：

```text
模型自动标注 -> 人工修正/确认 -> 进入训练集
```

### 1.3 先稳定识别，再增强策略

识别不稳定时，策略越复杂越容易误导用户。

开发优先级：

```text
区域归属稳定性 > 牌局阶段识别 > 回放测试 > 基础策略 > 高级策略
```

### 1.4 不直接宣传为“最优出牌 AI”

现阶段项目更准确的定位是：

```text
血流麻将实时识别与出牌教学辅助系统
```

等后续加入完整番型、风险判断、碰杠收益、多轮模拟后，再考虑升级为“强 AI 辅助”。

---

# 阶段 0：项目清理与主线统一

## 0.1 阶段目标

形成一个干净、可运行、可继续开发、可交付的项目主线。

当前包里发现了敏感文件和文档/配置不一致问题，需要先修复，否则后续开发会越来越乱。

## 0.2 必做任务

### 任务 0.2.1 删除敏感文件

检查并删除：

```text
IP账号密码.txt
```

如果该文件里是真实账号、密码、服务器 IP、Token、密钥，需要立刻作废或更换。

同时检查以下文件类型：

```text
*.env
*password*
*passwd*
*secret*
*token*
*账号*
*密码*
```

### 任务 0.2.2 统一默认模型路径

统一所有配置和文档中的默认模型路径。

当前主模型：

```text
models/yolo/xueliu_final325_clean_v1_0703.pt
```

需要检查并修改：

```text
README.md
configs/model.yaml
src/xueliu_ai/ui/realtime_app.py
src/xueliu_ai/main.py
models/yolo/training_rounds.yaml
```

旧路径示例：

```text
models/yolo/xueliu_tiles_v1.pt
```

应全部替换为当前主模型路径。

### 任务 0.2.3 重写 training_rounds.yaml

只保留 clean start 后的主线模型记录：

```text
manual39 -> manual113 -> manual222 -> final325
```

建议格式：

```yaml
current_model: models/yolo/xueliu_final325_clean_v1_0703.pt

rounds:
  - name: manual39_clean_v1
    model: models/yolo/xueliu_manual39_clean_v1.pt
    manual_images: 39
    source: clean_start_manual
    status: archived
    notes: 第一轮干净人工样本

  - name: manual113_clean_v2_0702
    model: models/yolo/xueliu_manual113_clean_v2_0702.pt
    manual_images: 113
    source: clean_start_manual
    status: archived
    notes: 第二轮人工核对样本

  - name: manual222_clean_v3_0703
    model: models/yolo/xueliu_manual222_clean_v3_0703.pt
    manual_images: 222
    source: clean_start_manual
    status: archived
    notes: 第三轮人工核对样本

  - name: final325_clean_v1_0703
    model: models/yolo/xueliu_final325_clean_v1_0703.pt
    manual_images: 325
    source: clean_start_manual
    status: current
    metrics:
      precision: 0.95806
      recall: 0.95960
      map50: 0.97367
      map50_95: 0.82143
    notes: 当前主模型
```

注意：

- 不要保留旧 Roboflow 主线。
- 不要保留伪标注主线。
- 不要写绝对路径，例如 `F:\xxx`、`G:\xxx`。
- 所有路径使用相对路径。

### 任务 0.2.4 拆分交付包

建议以后分成三个包：

```text
runtime_package
开发包 dev_package
证据包 evidence_package
```

#### runtime_package

只包含运行必需内容：

```text
src/
configs/
models/yolo/xueliu_final325_clean_v1_0703.pt
models/yolo/training_rounds.yaml
README.md
启动脚本
血流麻将实时辅助.exe
requirements.txt 或 pyproject.toml
```

不包含：

```text
data/
datasets/
runs/
reports/
.git/
.venv/
.env
账号密码文件
旧截图
旧 docx
旧训练数据
```

#### dev_package

包含开发需要的源码和测试：

```text
src/
tests/
configs/
scripts/
docs/
pyproject.toml
requirements.txt
```

#### evidence_package

保存证据材料和评估报告：

```text
reports/
回放截图/
评估表格/
模型训练曲线/
```

## 0.3 新增脚本：runtime 打包脚本

新增文件：

```text
scripts/package_runtime.py
```

目标：

```text
一键生成干净运行包
```

建议命令：

```powershell
python scripts/package_runtime.py --output dist/xueliu_runtime_20260705.zip
```

脚本要求：

- 自动排除敏感文件。
- 自动排除大体积训练数据。
- 自动检查主模型是否存在。
- 自动检查包里不能出现 `IP账号密码.txt`。
- 自动输出包大小和文件数量。

## 0.4 阶段验收标准

阶段 0 完成后，必须满足：

```text
1. 包里没有账号密码文件。
2. README、配置、UI 默认模型路径一致。
3. training_rounds.yaml 只保留 clean start 主线。
4. runtime 包可以独立解压运行。
5. 项目根目录没有明显的旧垃圾产物。
6. pytest 能通过。
```

## 0.5 给 Codex 的开发提示词

```text
你现在接手血流麻将 AI 辅助项目，先执行阶段 0：项目清理与主线统一。

目标：
1. 删除敏感文件，尤其是 IP账号密码.txt。
2. 检查并清理所有可能包含账号、密码、token、secret 的文件。
3. 统一默认模型路径为 models/yolo/xueliu_final325_clean_v1_0703.pt。
4. 修改 README.md、configs/model.yaml、UI 默认参数和 main.py 中的旧模型路径。
5. 重写 models/yolo/training_rounds.yaml，只保留 clean start 后 manual39、manual113、manual222、final325 四个阶段。
6. 新增 scripts/package_runtime.py，用于生成干净 runtime zip 包。
7. runtime 包不得包含 data、datasets、runs、reports、.git、.venv、.env、账号密码文件。
8. 修改完成后运行 pytest，确认测试通过。
9. 输出变更摘要、删除文件列表、修改文件列表、测试结果。

注意：
- 不要改业务逻辑。
- 不要训练模型。
- 不要引入新框架。
- 所有路径使用相对路径。
```

---

# 阶段 1：建立黄金视频回放测试集

## 1.1 阶段目标

建立固定的回放测试集，用来判断每次修改是否让系统变好或变差。

没有黄金测试集之前，不建议继续大规模改区域逻辑或训练新模型。

## 1.2 为什么要先做回放测试集

当前最大问题是实时牌局中的区域归属和阶段判断。

如果没有固定回归测试，每次修改都只能靠肉眼感觉：

```text
这次好像更稳了
这个视频好像可以
另一个视频又坏了
```

这样项目很难收敛。

必须变成：

```text
固定视频/固定图片 -> 自动测试 -> 输出指标 -> 对比上一次结果
```

## 1.3 黄金测试集目录建议

新增目录：

```text
data/gold_replay/
```

目录结构：

```text
data/gold_replay/
  images/
    000001_normal_hand_no_meld.png
    000002_drawn_tile_split.png
    000003_one_meld.png
    ...
  gold_cases.json
  README.md
```

## 1.4 gold_cases.json 格式

建议格式：

```json
[
  {
    "image": "images/000001_normal_hand_no_meld.png",
    "case_type": "normal_hand_no_meld",
    "phase": "MY_TURN",
    "allow_recommend": true,
    "expected_my_hand_count": 14,
    "expected_my_meld_count": 0,
    "expected_missing_suit_known": true,
    "notes": "普通出牌阶段，无碰杠"
  },
  {
    "image": "images/000002_deal_animation.png",
    "case_type": "deal_animation",
    "phase": "DEALING",
    "allow_recommend": false,
    "expected_my_hand_count": null,
    "expected_my_meld_count": null,
    "expected_missing_suit_known": false,
    "notes": "发牌动画中，不允许推荐"
  }
]
```

## 1.5 样本类型规划

第一批建议做 100～200 张，重点覆盖：

```text
normal_hand_no_meld          无碰杠普通手牌
drawn_tile_split             摸进单张分离
one_meld                     碰一组后
two_melds                    碰两组后
three_melds                  碰三组后
four_melds                   碰四组后
deal_animation               发牌中
exchange_three               换三张
choose_missing_suit          定缺
my_turn                      我的出牌阶段
other_turn                   他人出牌阶段
action_popup                 碰/杠/胡提示
settlement                   结算页
occlusion                    弹窗/遮挡
low_resolution               低清晰度投屏
different_window_size        不同窗口比例
```

## 1.6 新增命令：replay-test

新增 CLI 命令：

```powershell
python -m xueliu_ai replay-test --gold data/gold_replay/gold_cases.json
```

输出内容：

```text
总样本数
阶段识别准确率
手牌数量准确率
碰杠数量准确率
允许推荐准确率
禁止推荐准确率
失败样本列表
失败截图输出目录
```

失败输出目录：

```text
reports/replay_failures/YYYYMMDD_HHMMSS/
```

每个失败样本输出：

```text
原图
带识别框图
预测 JSON
错误原因
```

## 1.7 阶段验收标准

阶段 1 完成后，必须满足：

```text
1. 至少 100 张黄金测试图片。
2. 覆盖普通手牌、摸牌、碰杠、换三张、定缺、结算、弹窗等阶段。
3. replay-test 命令可以运行。
4. 每次测试输出指标。
5. 失败样本可以定位到具体图片。
6. 当前 baseline 指标被保存，作为后续对比基准。
```

## 1.8 给 Codex 的开发提示词

```text
执行阶段 1：建立黄金视频回放测试集和 replay-test 命令。

目标：
1. 新增 data/gold_replay 目录结构说明。
2. 定义 gold_cases.json 格式。
3. 新增 replay-test CLI 命令。
4. replay-test 输入 gold_cases.json，逐张图片运行当前 YOLO 检测、区域归属、阶段判断和推荐开关判断。
5. 输出阶段识别准确率、手牌数量准确率、碰杠数量准确率、allow_recommend 准确率。
6. 对失败样本输出 debug 图片和 JSON 到 reports/replay_failures/。
7. 当前没有足够图片时，先支持 gold_cases.json 中少量样本运行，不要硬编码路径。
8. 增加单元测试，验证 gold_cases.json 解析、指标统计、失败输出逻辑。

注意：
- 不要修改模型。
- 不要大改 UI。
- replay-test 应该复用现有 realtime_table 和检测逻辑。
- 代码要能在没有真实视频的情况下跑通测试。
```

---

# 阶段 2：实现牌局阶段状态机 GamePhase

## 2.1 阶段目标

让系统知道当前处于什么牌局阶段，并根据阶段决定是否允许出牌推荐。

当前不能只靠手牌数量判断是否可以推荐。

## 2.2 新增状态定义

建议新增文件：

```text
src/xueliu_ai/table/game_phase.py
```

状态枚举：

```python
from enum import Enum

class GamePhase(str, Enum):
    UNKNOWN = "UNKNOWN"
    LOADING = "LOADING"
    DEALING = "DEALING"
    EXCHANGE_THREE = "EXCHANGE_THREE"
    CHOOSE_MISSING_SUIT = "CHOOSE_MISSING_SUIT"
    WAITING = "WAITING"
    MY_TURN = "MY_TURN"
    ACTION_POPUP = "ACTION_POPUP"
    SETTLEMENT = "SETTLEMENT"
```

## 2.3 推荐开关规则

新增统一函数：

```python
def should_allow_recommend(context) -> tuple[bool, list[str]]:
    ...
```

推荐必须同时满足：

```text
phase == MY_TURN
手牌数量合法
连续稳定 N 帧
没有严重区域异常
没有弹窗遮挡
已经完成定缺
不是换三张
不是发牌中
不是结算页
```

不满足时，返回：

```text
allow_recommend = false
reasons = [...]
```

## 2.4 阶段识别初版规则

第一版可以不用 OCR，先基于视觉和牌数量规则。

### DEALING 发牌中

特征：

```text
手牌数量持续变化
手牌数量明显小于正常数量
底部牌未稳定
多帧检测结果波动大
```

### EXCHANGE_THREE 换三张

特征：

```text
手牌数量接近 13
存在选择三张相关 UI 区域
底部可能有选中牌状态
```

如果暂时无法识别 UI 文案，可以先保守处理：

```text
刚开局前若阶段不确定，禁止推荐
```

### CHOOSE_MISSING_SUIT 定缺

特征：

```text
开局阶段
出现三门花色选择 UI
定缺状态未知
```

如果无法识别 UI，先用人工配置或 UI 按钮确认定缺。

### MY_TURN 我的出牌阶段

特征：

```text
手牌数量为 14 / 11 / 8 / 5 / 2
检测稳定
没有弹窗
没有结算页
```

### WAITING 等待他人出牌

特征：

```text
手牌数量为 13 / 10 / 7 / 4 / 1
检测稳定
没有弹窗
```

### ACTION_POPUP 操作提示

特征：

```text
屏幕中存在碰、杠、胡、过等操作按钮区域
```

第一版可以只做 ROI 占位，后续再训练或模板识别。

### SETTLEMENT 结算页

特征：

```text
中间出现大面积结算面板
麻将牌检测数量明显下降
UI 布局不再是牌桌
```

## 2.5 UI 显示要求

实时 UI 右侧增加：

```text
当前阶段：MY_TURN
推荐状态：允许 / 禁止
禁止原因：
- 定缺未知
- 手牌数量不稳定
- 疑似过渡帧
```

预览窗口区分：

```text
当前帧检测结果
稳定后结果
策略使用结果
```

## 2.6 阶段验收标准

```text
1. 新增 GamePhase 枚举。
2. 新增 should_allow_recommend 统一推荐开关。
3. UI 能显示当前阶段、推荐状态、禁止原因。
4. replay-test 能评估 phase 和 allow_recommend。
5. 发牌、换三张、定缺、结算页默认禁止推荐。
6. 原有测试通过。
```

## 2.7 给 Codex 的开发提示词

```text
执行阶段 2：实现牌局阶段状态机 GamePhase。

目标：
1. 新增 GamePhase 枚举，包含 UNKNOWN、LOADING、DEALING、EXCHANGE_THREE、CHOOSE_MISSING_SUIT、WAITING、MY_TURN、ACTION_POPUP、SETTLEMENT。
2. 新增 phase_detector，用当前检测结果、区域结果、稳定帧信息推断当前阶段。
3. 新增 should_allow_recommend(context)，统一判断是否允许出牌推荐。
4. 推荐必须只在 MY_TURN 且手牌数量稳定合法时允许。
5. 发牌中、换三张、定缺中、操作弹窗、结算页、未知状态，一律禁止推荐。
6. UI 显示当前 phase、allow_recommend、禁止原因。
7. replay-test 接入 phase 和 allow_recommend 指标。
8. 增加单元测试覆盖各阶段基本判断。

注意：
- 第一版可以使用规则法，不需要 OCR。
- 不要删除原有稳定帧逻辑，应接入状态机。
- 禁止推荐时仍然显示识别结果，但不输出出牌建议。
```

---

# 阶段 3：增强实时区域归属

## 3.1 阶段目标

解决当前最核心问题：

```text
手牌区和碰杠区边界不稳定
右侧碰牌被误并入手牌
摸牌单张被误分区
碰杠后手牌数量变化导致判断错误
```

## 3.2 新的区域归属思路

不要直接在全桌上粗分区域，而是分两步。

### 第一步：识别我的底部操作区

定义：

```text
my_play_area = 屏幕底部属于我的完整牌区
```

这个区域包括：

```text
我的暗手牌
我的摸牌
我的碰牌
我的杠牌
```

### 第二步：在 my_play_area 内部再分类

分类为：

```text
concealed_hand      暗手牌
drawn_tile          摸进单张
my_meld             我的碰杠牌
uncertain           不确定
```

## 3.3 手牌数量约束

根据碰杠组数约束手牌数量：

```text
0 组碰杠：13 / 14 张
1 组碰杠：10 / 11 张
2 组碰杠：7 / 8 张
3 组碰杠：4 / 5 张
4 组碰杠：1 / 2 张
```

如果不符合，不要强行推荐，而是：

```text
allow_recommend = false
reason = 区域数量不合法，疑似过渡帧或区域误归属
```

## 3.4 摸牌单张识别

摸牌通常表现为：

```text
在暗手牌右侧
与主手牌列有明显间隔
y 坐标接近主手牌
大小接近主手牌
```

第一版规则：

```text
1. 找到底部同一排的主手牌簇。
2. 如果最右侧有一张与主簇间距明显大于平均牌宽，则标为 drawn_tile。
3. drawn_tile 仍然属于手牌，但 UI 上单独显示。
```

## 3.5 碰杠区识别

我的碰杠区通常在底部右侧或手牌旁边。

第一版规则：

```text
1. 根据牌的 y 坐标区分暗手牌行和碰杠牌行。
2. 根据牌间距和组合形态识别 3 张/4 张一组。
3. 如果某组牌明显不在暗手牌主行，归为 my_meld。
4. 如果不确定，进入 uncertain，不参与策略。
```

## 3.6 区域诊断输出

每帧输出区域诊断：

```json
{
  "my_hand_count": 14,
  "my_meld_tile_count": 0,
  "my_meld_group_count": 0,
  "drawn_tile_count": 1,
  "is_count_legal": true,
  "diagnostics": []
}
```

异常示例：

```json
{
  "my_hand_count": 12,
  "my_meld_tile_count": 3,
  "my_meld_group_count": 1,
  "is_count_legal": false,
  "diagnostics": [
    "1组碰杠时手牌应为10或11张，当前识别为12张",
    "疑似手牌尾部和碰杠区边界混淆"
  ]
}
```

## 3.7 阶段验收标准

```text
1. 我的手牌和我的碰杠区分离逻辑独立成模块。
2. 支持 drawn_tile 单独识别。
3. 支持 meld_group_count 计算。
4. 支持手牌数量合法性判断。
5. 区域异常时禁止推荐。
6. replay-test 中 one_meld、two_melds、drawn_tile_split 类别准确率明显提升。
```

## 3.8 给 Codex 的开发提示词

```text
执行阶段 3：增强实时区域归属，重点解决我的手牌、摸牌和我的碰杠区边界问题。

目标：
1. 将我的底部区域识别独立为 my_play_area。
2. 在 my_play_area 内部分类 concealed_hand、drawn_tile、my_meld、uncertain。
3. 新增 drawn_tile 识别规则：识别手牌右侧分离单张。
4. 新增 my_meld_group_count 计算，识别底部碰杠组数。
5. 根据碰杠组数约束合法手牌数：
   - 0组：13/14
   - 1组：10/11
   - 2组：7/8
   - 3组：4/5
   - 4组：1/2
6. 区域数量不合法时，禁止推荐，并输出诊断原因。
7. replay-test 增加区域诊断指标。
8. 增加单元测试覆盖无碰杠、一组碰杠、两组碰杠、摸牌单张分离等场景。

注意：
- 不要删除原有 realtime_table 功能，优先重构为兼容式改造。
- 不确定的牌不要强行归入手牌。
- 宁可禁止推荐，也不要基于错误手牌给建议。
```

---

# 阶段 4：补充 hard case 数据并训练新模型

## 4.1 阶段目标

在区域测试框架建立后，再补充真实投屏 hard case 样本，训练新模型。

下一版模型目标不是普通 mAP 更高，而是实际牌局回放更稳定。

## 4.2 数据补充重点

优先采集：

```text
碰杠后底部手牌缩短
右侧碰牌贴近手牌尾部
摸牌单张离主手牌列较远
左右旋转牌
180度对家牌
弃牌区密集堆叠
低清晰度投屏
窗口缩放变化
发牌动画
换三张界面
定缺界面
碰/杠/胡弹窗
结算页
```

## 4.3 数据规模建议

下一批建议补充：

```text
新增人工确认图片：200～300 张
总人工确认图片：500～650 张
```

命名建议：

```text
xueliu_final500_region_hardcases_v2.pt
```

或：

```text
xueliu_final650_realtime_hardcases_v2.pt
```

## 4.4 标注原则

### 必须人工确认

```text
所有进入训练集的图片和标签必须人工确认
```

### 可以使用模型预标注

流程：

```text
1. 用当前 final325 模型预标注
2. 人工打开 LabelU/CVAT 检查
3. 修错框、补漏框、删错框
4. 导出 YOLO 标签
5. 加入训练集
```

### 不要加入无关风格数据

暂时不要重新引入：

```text
Roboflow 杂数据
日麻数据
实物麻将数据
非当前游戏 UI 数据
```

## 4.5 新模型验收指标

不要只看 YOLO mAP。

必须同时比较：

```text
gold replay 阶段识别准确率
gold replay 手牌数量准确率
gold replay 碰杠归属准确率
gold replay allow_recommend 准确率
视频回放中推荐稳定性
```

新模型必须不低于旧模型：

```text
xueliu_final325_clean_v1_0703.pt
```

如果 mAP 更高但回放效果更差，不切主模型。

## 4.6 阶段验收标准

```text
1. 新增 200～300 张人工确认 hard case 样本。
2. 数据集版本号清晰。
3. 训练脚本可复现。
4. 新模型和旧模型在 replay-test 上对比。
5. 只有 replay-test 更好或至少不退化，才切换主模型。
6. 更新 training_rounds.yaml。
```

## 4.7 给 Codex 的开发提示词

```text
执行阶段 4：补充 hard case 数据并训练新模型。

目标：
1. 基于真实投屏视频抽取 hard case 图片。
2. 用当前 final325 模型进行预标注。
3. 生成待人工审核目录，不得直接进入训练集。
4. 设计数据集版本 xueliu_final500_region_hardcases_v2。
5. 编写训练配置和训练脚本。
6. 训练完成后，自动运行 replay-test，对比 final325 和新模型。
7. 输出模型对比报告，包括 mAP、手牌数量准确率、碰杠归属准确率、allow_recommend 准确率。
8. 只有新模型不退化，才建议更新 current_model。

注意：
- 不要使用未经人工确认的标签训练。
- 不要把外部风格不一致的数据重新混进主线。
- 如果新模型 mAP 提高但 replay-test 下降，不允许切换主模型。
```

---

# 阶段 5：升级血流麻将策略层

## 5.1 阶段目标

在识别稳定后，增强策略，让系统不只是“能推荐一张牌”，而是能解释为什么推荐。

## 5.2 策略升级顺序

建议顺序：

```text
1. 定缺推荐
2. 换三张推荐
3. 出牌推荐增强
4. 碰牌判断
5. 杠牌判断
6. 番型计算
7. 风险控制
8. 多轮模拟
```

## 5.3 定缺推荐

输入：

```text
当前 13 张手牌
三门花色数量
对子数量
搭子数量
有效进张
孤张数量
```

输出：

```text
推荐定缺：万 / 筒 / 条
理由：
- 该门数量最少
- 有效搭子最少
- 保留价值最低
```

第一版评分：

```text
花色分 = 数量 * 10 + 搭子数 * 8 + 对子数 * 6 + 高价值形状分
分数最低的花色优先定缺
```

## 5.4 换三张推荐

输入：

```text
当前 13 张手牌
每门花色结构
孤张
对子
搭子
定缺候选
```

输出：

```text
推荐换出三张
换牌理由
换后牌型评估
```

优先级：

```text
1. 换掉弱花色孤张
2. 避免拆强搭子
3. 尽量保留对子和连续搭子
4. 如果某门明显应该定缺，优先从该门换三张
```

## 5.5 出牌推荐增强

当前出牌建议主要看向听和有效牌。

下一步增加：

```text
牌型发展价值
定缺执行优先级
保留搭子价值
已见牌影响
危险牌风险
番型潜力
```

输出格式：

```text
推荐打：三万

理由：
1. 属于定缺花色，必须优先打出。
2. 打出后不增加向听。
3. 保留 4-5 筒、6-7 条两组有效搭子。
4. 当前三万已见 2 张，保留价值低。
```

## 5.6 碰牌判断

不要只判断“能不能碰”，要判断“该不该碰”。

判断维度：

```text
碰后向听是否下降
是否破坏潜在顺子
是否增加番型收益
是否暴露牌型
是否影响定缺
是否会导致手牌变差
```

输出：

```text
建议：不碰

理由：
- 碰后向听不变
- 会拆掉 3-4-5 的顺子潜力
- 当前该牌已见较少，继续摸进价值更高
```

## 5.7 杠牌判断

区分：

```text
暗杠
明杠
补杠
```

判断维度：

```text
收益：杠分、摸牌机会、番型提升
风险：点炮风险、暴露信息、被抢杠
阶段：早期/中期/后期
```

第一版可以保守：

```text
暗杠优先级较高
明杠看是否破坏牌型
补杠在高风险阶段谨慎
```

## 5.8 番型计算

逐步支持：

```text
平胡
对对胡
清一色
七对
龙七对
金钩钓
杠上花
根
```

先实现番型识别，再实现收益权重。

## 5.9 阶段验收标准

```text
1. 定缺推荐可用。
2. 换三张推荐可用。
3. 出牌理由更清楚。
4. 碰牌建议不是简单能碰就碰。
5. 杠牌建议区分暗杠、明杠、补杠。
6. UI 显示推荐理由。
7. 策略单元测试覆盖典型手牌。
```

## 5.10 给 Codex 的开发提示词

```text
执行阶段 5：升级血流麻将策略层。

目标：
1. 新增定缺推荐模块，根据三门数量、搭子、对子、有效进张判断推荐定缺。
2. 新增换三张推荐模块，输出建议换出的三张牌和理由。
3. 增强出牌推荐，不只输出牌名，还要输出推荐理由。
4. 重构碰牌建议，判断该不该碰，而不是只判断能不能碰。
5. 重构杠牌建议，区分暗杠、明杠、补杠，并输出收益风险说明。
6. 逐步增加番型识别：平胡、对对胡、清一色、七对、龙七对、金钩钓、根。
7. 所有策略模块增加单元测试。
8. UI 显示推荐理由和风险提示。

注意：
- 策略层必须只使用稳定后的手牌结果。
- 如果区域诊断异常，不允许输出策略。
- 不要在识别不稳定阶段强行推荐。
```

---

# 阶段 6：UI 和用户体验优化

## 6.1 阶段目标

让普通用户能看懂系统现在在做什么，为什么推荐或为什么不推荐。

## 6.2 UI 需要显示的信息

右侧面板建议分区：

```text
1. 当前阶段
2. 推荐状态
3. 我的手牌
4. 我的碰杠
5. 定缺状态
6. 推荐出牌
7. 推荐理由
8. 区域诊断
9. 模型状态
```

## 6.3 当前帧、稳定帧、策略帧区分

UI 上明确显示：

```text
当前帧检测：YOLO 当前画面识别结果
稳定后结果：连续多帧确认后的结果
策略使用结果：真正送入策略的手牌
```

避免用户看到预览框和右侧建议短暂不一致时误以为系统错误。

## 6.4 推荐状态显示

示例：

```text
当前阶段：发牌中
推荐状态：禁止推荐
原因：
- 手牌数量未稳定
- 疑似过渡帧
```

示例：

```text
当前阶段：我的出牌
推荐状态：允许
推荐打：九万
理由：
- 属于定缺花色
- 已见 2 张
- 打出后不增加向听
```

## 6.5 阶段验收标准

```text
1. UI 能显示 phase。
2. UI 能显示 allow_recommend。
3. UI 能显示禁止原因。
4. UI 能显示稳定后手牌和当前帧区别。
5. UI 能显示推荐理由。
6. 普通用户能判断系统当前是否可信。
```

---

# 阶段 7：最终交付与版本管理

## 7.1 版本命名

建议版本号：

```text
v0.1.0  当前可运行 MVP
v0.2.0  加入黄金回放测试
v0.3.0  加入 GamePhase 状态机
v0.4.0  优化区域归属
v0.5.0  新 hard case 模型
v0.6.0  策略增强
v0.7.0  UI 体验优化
v1.0.0  可交付稳定版
```

## 7.2 每个版本必须有 release note

格式：

```markdown
# v0.3.0 Release Note

## 新增
- 新增 GamePhase 状态机
- 新增推荐开关 should_allow_recommend

## 修改
- UI 显示当前阶段
- 策略只使用稳定手牌

## 修复
- 修复发牌过程中误推荐问题

## 测试
- pytest: xx passed
- replay-test: phase accuracy xx%
- allow_recommend accuracy xx%

## 已知问题
- 定缺界面仍无法稳定识别
```

## 7.3 最终 v1.0.0 验收标准

```text
1. runtime 包干净，无敏感文件。
2. 普通用户可以双击启动。
3. 支持屏幕 ROI 选择。
4. 实时识别稳定。
5. 发牌、换三张、定缺、结算页不误推荐。
6. 我的出牌阶段能稳定推荐。
7. 推荐理由可解释。
8. 至少 200 张黄金回放测试通过。
9. pytest 全部通过。
10. 有明确 README 和故障排查文档。
```

---

# 总执行顺序

建议严格按下面顺序推进：

```text
阶段 0：项目清理与主线统一
阶段 1：黄金视频回放测试集
阶段 2：GamePhase 牌局状态机
阶段 3：实时区域归属增强
阶段 4：hard case 数据补充与新模型训练
阶段 5：血流麻将策略层升级
阶段 6：UI 和用户体验优化
阶段 7：最终交付与版本管理
```

不要跳过阶段 1。  
不要在没有回放测试的情况下大改区域逻辑。  
不要在区域识别不稳定时过度投入高级策略。

---

# 当前最应该马上做的 10 件事

```text
1. 删除 IP账号密码.txt
2. 更换可能泄露的账号密码
3. 统一模型路径到 xueliu_final325_clean_v1_0703.pt
4. 重写 training_rounds.yaml
5. 写 package_runtime.py
6. 重新生成 runtime 包
7. 建立 data/gold_replay 目录
8. 设计 gold_cases.json
9. 实现 replay-test 命令
10. 把 GamePhase 接入 UI
```

---

# 给开发者的最终提示

这个项目不是失败项目，而是已经进入第二阶段的项目。

第一阶段解决的是：

```text
能不能识别麻将牌
```

第二阶段要解决的是：

```text
能不能稳定理解当前牌局状态
```

第三阶段才是：

```text
能不能给出更强的策略建议
```

所以现在不要急着继续训练，也不要急着做复杂 AI。  
先把状态机、回放测试和区域归属做扎实，后面的策略升级才有意义。
