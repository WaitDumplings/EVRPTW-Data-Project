# Stage 2 审阅与服务器复现

Stage 2 将一个已核验的城市 CLE 转换成 depot-day 实例。每个实例服务恰好 $N$ 个不同地点，每个地点访问一次并交付该地点的完整订单。不同日期与 $N$ 各自生成地点集合和订单。同一次运行共享选定的 depot、路网和来源版本。

首个服务器任务使用 [Dallas Cus100 审阅配置](../../examples/stage2/server_review.json)。将代码版本、输入版本和运行配置固定后，先核验一个实例，再决定是否扩至 Cus500、Cus1000、Cus2000 和其他城市。仓库提供构造和验证入口，真实输入通过独立的数据交接清单准备。

## 1. 输入和版本

| 输入 | 读取位置和身份 |
| --- | --- |
| Stage 1 CLE | `input.cle` 指向 `<stage1-output>/cities/<city>`。保留城市 manifest、独立核验、canonical locations、道路接入、速度和充电设备，以及项目根的 `stage1_server_build_receipt.json`。按 [Stage 1 构造与交接说明](../stage1/SERVER_AGENT_HANDOFF.zh-CN.md) 完成冻结 cohort 的构造与核验。 |
| Census 社区 | `input.communities` 指向本城 TIGER/Line 2025 block group GeoParquet，保留 CRS、geometry 和字符串 GEOID。tract 使用 12 位 BG GEOID 的前 11 位。 |
| Amazon donor | `input.donor_artifact` 指向 `prepare-marks` 构造的 GEN-TRAIN 订单属性库。输入是现有 Amazon Challenge training 的 `package_data.json` 与 `route_data.json` 原始字节。 |

新版 CLE 的 customer 是已准入且道路可用的 canonical building，住宅户数和 `units_source` 来自 Stage 1 的最终判定。保留 NSI 代表和邻居估计的区分。历史 CLE 保留原有冻结证书和目录结构，只用于同版本复现。

## 2. 生成顺序

1. **配送范围与 tract。** 审阅配置先在含可用 depot 的 BG 中等概率抽一个，再在该 BG 内等概率选 depot。depot 去客户的最快参考路径须在 5,445 秒内且满电可达，同时要求有道路回程。去程时间和能耗取同一路径。以活跃概率和 $\sum_i \pi_i$ 为 tract 抽样权重，沿道路支持的邻接逐步加入完整 tract。也允许从未选的非前沿 tract 新开片区。审阅配置把候选池目标设为 $20N$，上限是实际可服务地点数。池固定后使用精确条件 Bernoulli 抽取 $N$ 个地点。
2. **活跃户和完整订单。** 对住宅户数为 $u_i$ 的地点，活跃概率为 $\pi_i=1-(1-p)^{u_i}$。选定地点集合后，逐地点从条件于 $A_i>0$ 的二项分布抽活跃户数，再按 $K_i=A_i+\operatorname{NB}(A_i,1/\mu)$ 抽包裹件数，其中 NB 计失败次数。用 GEN-TRAIN donor 生成联合包裹属性。单地点完整包裹体积超过 18.5 m³ 时重提议整份订单，最多 64 次，耗尽后保留不支持记录和原因。
3. **一次访问的时间属性。** 每个地点共用一个时间窗，截止时刻表示最晚开始服务。服务时间按包裹数曲线、donor 残差和活跃户数的批量修正计算。默认班次 08:00–24:00，实际回仓仍受班次结束约束。
4. **道路成本与充电站。** 从冻结 CLE 取同一路网上的最短距离和最快时间两套有向路径成本，分别保持路径内的距离、时间和能耗对应。先绑定车辆兼容、已接入路网的候选站池，再按站点预算选站。
5. **保存、核验与导出。** 保存 depot、活跃 customer 和 CS 的 canonical ID、选区和订单记录、两套矩阵与来源身份。独立核验保存结果后再导出。验证通过表示构造规则和文件一致，车队路线可行性由后续求解实验验证。

车辆配置为同质 EV van，18.5 m³ 容量，100 kWh 电池，满电出发，10% 保留电量，车辆数无固定上限。单位里程能耗取当前 reference vehicle 设定。充电按有效功率线性满充，默认班次内可用，按当前无排队模型计算。客户和车辆保留同一套实例身份，默认目标为总道路距离。

5,445 秒沿用 Amazon GEN-TRAIN 的历史 depot 到 dropoff 时间参考上限，表示实例的道路准入条件。起始新开概率 0.1 和候选倍数 20 属于设计参数。随候选池填充，新开概率线性下降。无法映射到 tract 的道路可服务地点单列计数。

### 联合充电站选取

审阅配置启用 `tract_joint_budget`。Cus100、Cus500、Cus1000、Cus2000 分别取 20、30、40、60 个外部站，depot 单独计。配置可按规模修改 `external_cs_budgets_by_size`，其余规模使用 `external_cs_budget`。

- **先跨 tract。** 每个活跃 tract 选一个道路 medoid，以双向道路距离构造参考 MST，按两种方向分别评价每条边。站点候选是兼容公共站池。对可在参考电量下到站且满充后可到终点的候选，比较额外道路公里和行驶加满充分钟。最近、最快站给出归一化锚点，以 `1 / (1 + max(距离相对损失, 时间相对损失))` 计分。每轮选择能让所有参考连接的当前最佳分数平均提高最多的站。
- **再做片区内。** 从已选站起步，以客户到站距离和到站加满充时间继续评分。纯距离贪心选站集作为最近距离的保护基线。前两步预留尚需的基线席位，共享已选站。若联合评分结束后仍有基线差距，优先补入相应站点。
- **最后补足预算。** 比较客户到第二近公共站的平均道路距离。并列时依次考虑候选站贴近活跃 customer 的程度、有效功率和站点 ID。候选池不足时保存实际站数和不支持原因。

满充评分使用固定参考出发 SOC、到站道路能耗和有效功率。行驶时间取评分所用最短距离路径的时间。有效功率为站点来源功率与车辆上限的较小值，再乘 0.9。审阅配置取 `reference_departure_soc=0.8`，`0.5` 是已试算的敏感性对照。参考 SOC 是生成时的评分假设，求解时充电量由实际到站 SOC 决定。MST 提供选站参考联系，车辆访问顺序由求解器确定。

每个实例的 `charging_selection.json` 记录预算、候选站数、入选 ID、选取阶段和逐轮增益。选站记录中的 `between_joint`、`local_joint`、`local_protection` 和 `padding` 分别对应跨 tract、片区内、最近距离保障和补足预算。

## 3. 服务器复现顺序

从审阅通过的 Git commit 或字节一致的源码包安装，固定外部输入目录。以下命令在仓库根目录执行。替换文件路径，运行配置和输出放在服务器独立目录。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[osm]'

evrptw-stage2 prepare-marks \
  --packages /data/evrptw/amazon/raw/package_data.json \
  --routes /data/evrptw/amazon/raw/route_data.json \
  --output /data/evrptw/amazon/joint-marks \
  --partition training

mkdir -p /data/evrptw/configs
cp examples/stage2/server_review.json /data/evrptw/configs/dallas-cus100.json
# Edit the input and output paths before running.
evrptw-stage2 plan /data/evrptw/configs/dallas-cus100.json
evrptw-stage2 build /data/evrptw/configs/dallas-cus100.json
evrptw-stage2 verify /data/evrptw/runs/dallas-cus100-review --check-sources
evrptw-stage2 inspect /data/evrptw/runs/dallas-cus100-review
```

完整来源核验会读取 donor manifest 指向的原始 JSON。首次在服务器执行 `prepare-marks` 使用新目录，让来源路径指向服务器实际位置。保存原始 JSON 的字节、既有分区及来源哈希。配置相对路径以配置文件所在目录为准。每次正式执行使用新输出目录，失败时保留 `.partial` 或核验报告。

用 `inspect` 输出的 `view_id` 导出可供求解器读取的实例：

```bash
mkdir -p /data/evrptw/exports
evrptw-stage2 export /data/evrptw/runs/dallas-cus100-review VIEW_ID \
  --objective total_distance --path-policy shortest_distance \
  --output /data/evrptw/exports/dallas-cus100.json
```

首个任务的交付包括配置、输入和代码身份、`manifest.json`、`verification.json`、对应实例的选站记录与输出位置、构造和验证日志、耗时和峰值内存。验收查看恰好 $N$ 个不同 customer、实际外部站数、矩阵路径策略与来源绑定，并单列不支持的 workload 或候选池不足。扩到多规模时设置 `instances.sizes=[100,500,1000,2000]`，保留需要构造密集矩阵的 `instances.materialize_day_indices`，按首任务的资源记录确定并发。最快时间准入按工作日或周末速度表构造，一次运行使用同一类日型。需要两种日型时，分别运行。

## 4. 审阅证据的边界

- 本机按归档 Dallas 路网和同一 depot 做过四规模选站试算。以 80% 参考 SOC 的 Cus1000 为例，前五座联合评分站将跨 tract 平均最快补能时间从 212.0 分钟降到 35.5 分钟，同一组站能提供的最小绕行均值从 3.86 km 增到 4.65 km。两项分别取该站组可达到的最优值，可以来自不同的站。该批四个样本补齐固定站数后的最近距离、平均最快补能时间和 DC 数，与纯距离对照相同。
- $p=0.03$、$\mu=1.4$ 是按 Dallas 与 LA 的楼栋户数、固定 $N$ 机制及 Amazon GEN-TRAIN 标记得到的模型默认值。拟合时新版 CLE 还未施加正式道路接入，$p$ 不代表观测到的全国住户日收件率。候选池倍数 $\rho=20$ 是已确定的设计值。
- 老版 demand-first 配置仍可由 `evrptw-stage2 init --legacy` 生成，旧 schema 使用 `views.*` 及嵌套前缀，见[旧版配置](../../examples/stage2/legacy_natural_day.json)。原 territory-first 的 BG 时钟增长和绕行选站配置也保留。当前审阅任务从 [server_review.json](../../examples/stage2/server_review.json) 显式启用新策略。
- 服务器新 CLE 接入、11 城批量生成和路线求解验收属于后续交付。训练和测试 split、出发地配额及其支持率属于批量数据集层的验收。
