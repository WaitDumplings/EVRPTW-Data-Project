# Stage 1 冻结构造版本：服务器 Agent 任务书

本轮任务是使用 **`stage1-20261102-r1`** 的代码、来源和构造规则，在服务器生成 11 城完整 CLE，交付实际结果、日志和 manifest。版本身份以 `freeze/20261102-r1/MANIFEST.json` 为准。

本机已完成当前 11 城建筑属性重算：固定候选 **4,012,462** 栋，最终准入 **3,922,741** 栋，覆盖率 **97.7639%**；其中 NSI 代表 3,194,252、邻居估计 728,489。该结果经过最终面积/楼层规则，未解决 89,721 栋。它是道路接入前的建筑属性结果。服务器本轮还要完成道路、速度、三类地点接入、CLE 组装与实际验收；本交接不预先声明 Linux 全量运行通过。

## 1. 交接材料与执行范围

| 材料 | 用途 |
|---|---|
| `dist/stage1-20261102-r1.tar.gz` | 冻结构造代码、配置和方法文档 |
| `freeze/20261102-r1/MANIFEST.json` | 冻结 ID、代码/配置身份及归档记录 |
| `freeze/20261102-r1/input_manifest.json` | 当前真实输入的位置、来源和哈希记录 |
| `freeze/20261102-r1/stage1.server.json` | 11 城服务器配置模板；按实际路径改写 |
| `freeze/20261102-r1/run_stage1.sh` | 全 11 城分阶段执行入口 |

可调整服务器路径、输出位置、环境和资源线程数。保持原始数据字节、版本、城市集合、CRS、方法、种子和阈值不变；不增加来源、不逐点手工修数据、不调用 AI 生成建筑属性。本轮只构造 Stage 1。

固定城市为：`new-york`、`los-angeles`、`chicago`、`houston`、`phoenix`、`philadelphia`、`san-antonio`、`san-diego`、`dallas`、`jacksonville`、`fort-worth`。

## 2. 服务器准备

归档顶层为 `EVRPTW-Stage1-stage1-20261102-r1/`。例如：

```bash
mkdir -p /data/evrptw
tar -xzf /path/to/stage1-20261102-r1.tar.gz -C /data/evrptw
cd /data/evrptw/EVRPTW-Stage1-stage1-20261102-r1
```

在该项目根目录创建环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[osm]'
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

根据机器资源调整线程；开始使用逐城执行。记录实际 Python 与依赖环境。此处是服务器安装流程，安装和运行结果由本轮日志确认。本机实际环境保存在 `freeze/20261102-r1/environment.json`，`requirements-tested.txt` 是已观察版本记录，不是 Linux 二进制锁。

从 `stage1.server.json` 映射以下真实输入。路径可变，来源版本不变：

| 输入 | 保持的内容 |
|---|---|
| Microsoft Global | 2026-08-13 索引快照的真实建筑 polygon、ID 和可用独立高度 |
| OSM building / depot / roads | 2026-10-25T21:21:21Z 共同快照；完整建筑标签、设施证据、道路 ways/引用节点和原生访问限制 |
| NSI | 2026-10-26 归档的 NSI 2026 原始点和属性；保留零户数与正户数原始证据 |
| 边界 | 现有 TIGER/Line 2025 行政边界和扣水服务面；保留本城米制 CRS |
| AFDC | 当前 station、equipment、地址辅助归档及原生来源绑定；station 与 equipment 的各自采集时间分别保存 |
| NEI / StreetLight | 当前七州冻结速度 profile 和所绑定的 NEI 2023、商业车小时/星期权重及来源记录 |
| 来源 inventory | 现有文件键、输入哈希、来源时间、实际本地路径与设施/道路映射 |

当前输入清单为 116 个实际文件，约 2.614 GiB，包含完整道路 ways；按清单迁移这些输入即可，不需要 PBF 本体或作者旧 outputs 目录。完整字段与时间见[输入文档](../INPUTS.md)。代码包不代替这些原始地理数据。若数据从本机迁移，核对 `input_manifest.json` 的对应身份；缺少源文件时先补齐同一冻结来源，不以合成数据代替。`geometry_cache` 仅按交接配置所声明的导入模式使用，不能把缓存导入写成 raw 重建。

保留冻结模板和 inventory 的字节不变；需要调整路径时，在同目录复制一个运行配置：

```bash
python scripts/verify_stage1_freeze.py
cp freeze/20261102-r1/stage1.server.json freeze/20261102-r1/stage1.local.json
```

只编辑 `stage1.local.json` 中的输入／输出路径。inventory 内的相对文件层级保持不变，通过 `components.source_root` 指定实际数据根目录。默认模板已与输入布局一致，路径无需调整时可直接使用模板。下面命令使用复制出的运行配置：

```bash
evrptw-cle stage1 plan freeze/20261102-r1/stage1.local.json --json
```

`plan` 显示建筑与服务域输入状态；完整道路/设施输入按冻结 inventory 绑定。

## 3. 全 11 城统一校准，再逐城构造

**11 城只运行一次联合 `calibrate-units`。** 它逐城拟合 RMS 标度，再汇总校准数据产生共同分层门槛，写入 `<output>/units/quality_thresholds.json`。不要循环调用单城 `run` 或单城 `calibrate-units`；那会覆盖同一个全局门槛文件，并改变其他城市的合并回退依据。

`prepare` 内部已调用服务域 `scope`。正式顺序为：全部准备 → 全部一次校准 → 户数估计 → 全部定稿汇总 → 逐城组件/组装 → 全部验证。`finalize` 使用全 11 城调用，以保留完整项目汇总表。

配置路径改好后，可以直接运行交接脚本：

```bash
bash freeze/20261102-r1/run_stage1.sh freeze/20261102-r1/stage1.local.json
```

省略参数时使用脚本同目录的 `stage1.server.json`。断点入口 `STAGE1_FROM` 可选 `prepare`、`calibrate`、`estimate`、`finalize`、`components`、`verify`；默认 `prepare`。例如已经完成全部建筑定稿后：

```bash
STAGE1_FROM=components bash freeze/20261102-r1/run_stage1.sh freeze/20261102-r1/stage1.local.json
```

也可以执行下面的 Bash 展开流程。工作目录为项目根，先把 `RUN_OUTPUT` 设为配置中 `output` 的绝对路径：

```bash
set -euo pipefail
CONFIG="$PWD/freeze/20261102-r1/stage1.local.json"
RUN_OUTPUT="/data/evrptw/stage1-20261102-r1-output"
CITIES=(new-york los-angeles chicago houston phoenix philadelphia san-antonio san-diego dallas jacksonville fort-worth)
mkdir -p "$RUN_OUTPUT/logs"

evrptw-cle stage1 prepare "$CONFIG" --cities "${CITIES[@]}" --json 2>&1 \
  | tee "$RUN_OUTPUT/logs/01_prepare.json.log"

evrptw-cle stage1 calibrate-units "$CONFIG" --cities "${CITIES[@]}" --json 2>&1 \
  | tee "$RUN_OUTPUT/logs/02_calibrate_units.json.log"

evrptw-cle stage1 estimate-units "$CONFIG" --cities "${CITIES[@]}" --json 2>&1 \
  | tee "$RUN_OUTPUT/logs/03_estimate_units.json.log"

evrptw-cle stage1 finalize "$CONFIG" --cities "${CITIES[@]}" --json 2>&1 \
  | tee "$RUN_OUTPUT/logs/04_finalize.json.log"

COMPONENT_CITIES=(dallas new-york los-angeles chicago houston phoenix philadelphia san-antonio san-diego jacksonville fort-worth)
for CITY in "${COMPONENT_CITIES[@]}"; do
  evrptw-cle stage1 components "$CONFIG" --cities "$CITY" --component all --json 2>&1 \
    | tee "$RUN_OUTPUT/logs/05_components_${CITY}.json.log"
done

evrptw-cle stage1 verify "$CONFIG" --cities "${CITIES[@]}" --json 2>&1 \
  | tee "$RUN_OUTPUT/logs/06_verify.json.log"
```

上述参数不改变冻结规则。分阶段日志同时保留实际 stderr；使用服务器任务调度器时，将其保存到同一运行记录。若需要控制内存，可在联合校准结束后逐城调用 `estimate-units`，随后仍以全部城市调用一次 `finalize`。

脚本和展开流程使用同一阶段顺序。组件建议先串行完成 Dallas 的完整交接，再按资源处理其他城市；当前脚本默认串行，不自动分配城市并发。独立并行 components 时每个城市只交给一个进程，最终仍以全部城市执行 verify。恢复时保留已完成结果与原共同门槛，从失败阶段继续；不因某城失败而重选来源、放宽阈值或改写个别建筑。

## 4. 保持的关键规则

Customer 沿固定 MS–OSM 建筑分母构造。每条 NSI、每个真实 footprint 来源组和每栋最多选择一次，使用代表记录而不累加户数。原始 NSI 不覆盖。邻居模型只用原始训练参考和独立建筑特征；最终 `units_source` 明确为 `nsi`、`neighbor` 或 `unresolved`。

最终 U>1 必须有可评估且通过的面积/楼层宽松规模代理；一户来源支持保留既定例外。原生冲突、原始规模异常、邻居不支持和多户规模未评估保留原原因与分母。不能为提高覆盖率绕过准入规则。

路网保留 OSM 单行方向、机动车/配送禁令、锁闭和实际障碍；普通 private 按冻结配送穿行假设处理。Customer、depot、charging 与速度必须引用同一构造路网及原生有向边。Customer 接入使用完整 canonical polygon，depot 保留园区成员，charging 保留 AFDC 原点与设备来源。

完整方法见[五部分规格](../STAGE1_SPEC.zh-CN.md)；服务器本轮执行而不调参。

## 5. 完成交付

服务器 Agent 返回以下实际产物位置：

1. 冻结 ID、归档/代码身份、实际服务器配置、输入 manifest 和运行日志。
2. 项目级 `customer_coverage.csv`、`customer_uncovered_reasons.csv`、`units/` 标度与共同门槛，以及每城 `customer_inputs/` 的最终建筑和来源 ledger。
3. 11 城 `roads/`、`speeds/`、`customers/`、`depots/`、`charging/`、统一地点/接入表和 `cle_manifest.json`。
4. 全部城市的 `stage1_validation.json` 和项目 `validation_summary.json`，并列出每城实际属性准入、道路后可用 customer/depot/charging 数量。

验收读取保存的数据与 manifest：来源和代码身份正确；户数来源、最终规模准入和数量分账一致；三类地点/速度绑定共同有向图；11 城实际 validation 均 `passed=true`。不能只凭命令退出 0 或生成了目录宣布完成。

本机 97.7639% 作为已记录的建筑属性基线。服务器结果按真实运行输出，与同口径表核对；若出现差异，定位到输入、建筑范围、参考、户数选择或道路接入层，返回差异与日志，不为匹配数值修改冻结方法。完整 CLE 以本轮实际组件和验收记录为交付依据。

## 冻结身份与实际运行记录

本版身份以 `freeze/20261102-r1/MANIFEST.json` 的逐文件 SHA-256 和压缩包 SHA-256 为准。它包含本机已整理的工作区代码，不依赖 GitHub 默认分支的旧 HEAD。服务器使用本交接包，不从其他分支混入实现。

交接脚本先核对代码、实际配置引用的116个数据文件和固定方法参数；完成后保存 `<output>/stage1_server_build_receipt.json`，记录冻结清单、运行配置和11城CLE manifest的哈希。验证失败以非零退出，不凭日志存在判断完成。
