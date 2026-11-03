# Stage 1：城市配送环境（CLE）构造方法

版本：`stage1_building_units_rms_20261102`。按五个处理对象介绍：**路网 → 速度 → customer building with units → depot → charging station**。

Stage 1 构造固定地理环境、原始/估计属性及有向道路接入。Customer 是建筑地点，不做 apartment 小区合并。原始 NSI 不覆盖，多个来源的户数不相加；缺失户数使用数据标度和支持门槛校准后的邻居模型。实例地点激活、包裹、TW、service time 和车队属于 Stage 2。

- [运行与配置](stage1/SERVER_IMPLEMENTATION.zh-CN.md)
- [Customer 完整算法](stage1/CUSTOMER_METHOD.zh-CN.md)
- [Depot/charging 细节](stage1/DEPOT_CHARGING.zh-CN.md)
- [当前 11 城属性结果与统计口径](stage1/STATISTICS.zh-CN.md)

入口为 `evrptw-cle stage1`，等价入口为 `evrptw-data stage1`、`evrptw-stage1`。简单路径 `run` 自动串起准备、范围、户数校准/估计、定稿与五层构造；每步也可独立执行。

### 来源与年份

固定城市为New York、Los Angeles、Chicago、Houston、Phoenix、Philadelphia、San Antonio、San Diego、Dallas、Jacksonville、Fort Worth。城市选择参考Census Vintage2025、2025-07-01人口，以city proper固定cohort。

| 来源 | 当前方法使用内容/版本 | 时间含义 |
|---|---|---|
| Census TIGER/Line2025 Places、Area Hydrography | 行政面、扣水后的服务面 | 2025版边界 |
| Geofabrik OSM PBF | 道路、访问标签、building几何/用途、物流设施 | 同快照2026-10-25T21:21:21Z；对象编辑时间及PBF SHA另存 |
| Microsoft GlobalMLBuildingFootprints | 2026-08-13索引快照，主要建筑轮廓 | 索引日期不等于逐栋影像日期；不使用legacy |
| NSI2026公开API | 2026-10-26归档的点、住宅属性和源footprint线索 | 上游年份混合，resunits是模型估计 |
| AFDC station | 2026-10-26 23:28:25 UTC公共运行站点库存 | 库存采集时间，不等于安装年份 |
| AFDC charging-unit CSV | 2026-10-27 00:03:57 UTC接口、端口、功率 | 与station分别采集；洛杉矶当地仍10月26日 |
| Census Geocoder | 2026-10-26 23:47:41 UTC Address辅助响应 | Public_AR_Current；不覆盖GPS/Point原生坐标 |
| EPA2023 NEI最终MOVES CDB | source32速度分布 | 原始StreetLight观测2022-01至2023-05，含EPA补缺；最终处理主要2025年 |
| EPA2022v2 StreetLight COMM-MD temporal | 小时/星期VMT权重 | 实际商业车观测2022-11至2023-05；表处理2025-04，归档2025-09 |
| NEI2023月VMT | 年度代表日型加权 | 活动年2023；表版本2025-09，归档2025-10 |

分别保存 observation、model、release、acquisition 时间；逐栋影像年未知时保留未知。全部城市使用相同核心来源组合和方法。

## 1. 路网构造


### 1.1 城市、服务面与道路选择域

行政边界采用Census TIGER/Line 2025 city proper polygon。潜在地点的服务范围为行政区扣除同版水域。城市路网的选择包络为行政边界向外扩展20km的美国境内范围，城外道路用于通行、绕行和地点接入。

地点按geo_assignment_point确定城市归属；customer使用canonical building内部代表点，depot使用canonical成员代表点，CS使用AFDC原生点。完整来源几何及成员另存。跨界对象按代表点归属；建筑范围与NSI属性作用范围不一致时，原始户数另存，栋级赋值标为未决。

20km为已确认、可配置的路网选择域参数，不是逐路段统计估计出的阈值。OSM来源覆盖该包络，跨州区域采用相同2026-10-25T21:21:21Z快照的区域文件。按照原生OSM身份和对象版本合并重复来源，记录父PBF与派生提取的SHA。地理坐标保存为WGS84，距离、面积和投影在本城米制CRS计算。

### 1.2 原生道路、有向拓扑与访问规则

从OSM PBF选取与道路包络相交的highway ways，保存完整几何、原生节点序列、全部原始标签、对象版本和编辑时间。第二遍扫描保留所选完整ways引用的访问限制与障碍节点，包括位于选择包络之外的引用节点。

在真实共享节点、道路端点及访问限制节点处分段；中间曲线点保留在道路几何中。每条物理段按允许方向建立有向边，并保留平行边。图节点表示道路拓扑连接，地点通过接入表连接到图。

方向按原way语义解析：oneway=yes/true/1为原向，-1/reverse为反向，no/false/0为双向；缺省motorway和roundabout/circular采用原向，其他支持道路采用双向。未解析方向保留来源状态。路段长度沿完整道路折线计算，边身份包含源way和分段位置。[参考：OSM oneway](https://wiki.openstreetmap.org/wiki/Key:oneway)

路由车辆为轻型EV van的通用机动车代理。静态访问规则使用车型访问、配送用途、锁闭及物理barrier标签。车型访问优先级为motorcar、motor_vehicle、vehicle、access；配送显式禁令独立生效。

| 访问情况 | 路由处理 |
|---|---|
| 支持道路类上的缺失、yes、permissive、适用车型designated | 配送穿行 |
| 普通private道路及节点 | 配送穿行，允许shortcut；保存private_through_routing_assumed |
| destination、customers、delivery等目的限定道路 | 对应用途接入 |
| residents | 受邀配送接入，保存授权假设标记 |
| 明确机动车/配送禁令、locked、车辆不可通过的barrier | 对相应用途标为不可通行 |
| permit、military、agricultural、forestry等条件用途 | 按来源中的适用授权或配送例外求值 |
| 未解析访问取值 | 保留原始证据与pending状态 |

普通private穿行是本配送路网的统一建模假设。原始private属性继续保存。明确的方向、禁令、锁闭和物理障碍与该假设共同作用。设施类别、name/operator和机构代码不产生空间排除；实际对象的访问限制按其原生点、线、面范围作用。[参考：OSM access](https://wiki.openstreetmap.org/wiki/Key:access)

motorway、trunk及其links用于途经路由。地点接入承接道路采用版本化host-road profile：primary、secondary、tertiary及其links、unclassified、residential、living_street、service和road。construction、proposed、raceway、track保留在来源库存及其状态中。

### 1.3 配送路由图与地点接入

配送路由图G^{route}包含普通机动车可通行道路和按模型允许穿行的普通private道路。先按城市节点覆盖选择主要弱连通分量，再选择其中节点数最大的有向强连通分量作为reference SCC，并按稳定ID处理并列。

地点的出场与进场分别连接到该reference SCC。剩余目的限定道路及核心之外的真实接入段保存为delivery/charging用途接入路径。进、出gateway可以不同。普通private在路由图中直接参与最短路，位于同一private网络内的核心地点采用内部道路路径。

接入包括两部分：connector为来源几何到道路anchor的几何连接段；terminal为需要时从anchor到reference SCC的真实有向OSM路径。anchor位于道路中间时，保存方向偏移和部分边成本。公共充电站的地点资格和访问条件另按AFDC记录确定。

Customer用完整canonical建筑轮廓与道路求最近点对；depot用内部代表点并保留原生extent；CS用AFDC原生坐标。道路投影先检查最近物理道路及适用权限、方向和几何障碍。建筑阻挡或有向接入未确认时，按geometry gap、OSM way ID、segment ID的顺序检查局部替代候选。

局部搜索最多检查8条不同way；有来源extent时尺度为初始gap加两倍extent包围盒对角线，无extent时为两倍初始gap。候选使用同一套水域、其他建筑、受限制道路穿越及实际禁入对象判断。接入失败保留地点、来源几何、户数和原因。距离作为记录和诊断量保存。

输出包括原生物理道路库存、配送路由图、用途接入图、reference SCC、gateway、进出路径、部分边及connector信息。它们共同作为速度赋值与后续OD计算的输入。


### 1.4 输出与代码落实

输出边界、原生way库存、物理段、共同有向图、reference SCC及各用途接入。速度和三类地点均引用相同原生有向边身份。source way、物理段、有向边、SCC节点分别统计；服务器生成当前版本数量。

现有代码可复用[完整way提取](../scripts/extract_frozen_access_roads.py)、[物理/方向构造](../src/evrptw_cle/local_road_access.py)、[用途接入](../src/evrptw_cle/delivery_access.py)和[真实terminal路径](../src/evrptw_cle/terminal_access.py)。构造、charging、组装和reader共同使用本节访问规则；具体调用顺序见[运行说明](stage1/SERVER_IMPLEMENTATION.zh-CN.md)。

当前图是原生方向/访问标签支持的静态有向图。完整转向限制relations和逐时conditional规则的适用边界，在运行profile中明确记录。


## 2. 速度匹配


### 2.1 设计目标与来源分工

速度层为每条真实有向边配置限速、工作日/周末参考运行速度和通行时间。OSM提供逐边道路类型、方向和限速；EPA商业车辆统计提供州级、粗道路类相关的运行水平。

为保留逐边速度尺度，采用限速乘相对运行系数的结构。该结构参照限速缩放的道路速度模型[本节参考4]；本项目从EPA有效速度与低流量参考速度的比值构造系数。模型假设同州、同粗道路类共享相对运行比例，以OSM逐边限速表达细粒度道路差异。

### 2.2 数据来源与时间

| 输入 | 使用内容 | 版本与年份 |
|---|---|---|
| OSM PBF | 原生有向边、highway、通用及方向限速 | 2026-10-25T21:21:21Z快照 |
| 2023 NEI最终MOVES CDB | sourceType32的AvgSpeedDistribution及速度档位 | 底层StreetLight观测2022-01至2023-05，包含EPA补缺 |
| EPA 2022v2公开SMOKE temporal表 | 小时及星期VMT份额 | 商业车有效小时/日观测主要为2022-11至2023-05 |
| 2023 NEI活动表 | 州、月份、道路类及sourceType32的VMT | 2023月VMT用于年度代表工作日/周末聚合 |

sourceTypeID32（Light Commercial Truck）作为EV van的商业车辆统计代理。冻结输入采用EPA处理的StreetLight COMM-MD州级profile。源表、归档成员、SHA、转换SQL和参数保存在州profile配置中[本节参考1–2]。

工作日定义为周一至周五，周末为周六至周日。以2023月VMT和原始星期份额聚合七日小时分布，得到两种日型的小时VMT权重。每个CLE使用城市所属州的profile，包含该CLE的城外道路。

### 2.3 道路分类与OSM映射

| OSM highway | 采用的MOVES类别 |
|---|---|
| motorway、motorway_link、trunk、trunk_link | road4：urban restricted-access |
| 其他已纳入机动车路网的道路 | road5：urban unrestricted-access |

这里的restricted-access表示道路出入口控制类型。城市参考模型采用urban4/5；OSM trunk按road4代理归类。primary、secondary、tertiary、residential和service属于road5并共享该州、该日型的系数。它们的绝对速度由各自限速确定。普通private沿用本身highway及限速配置速度。

H/M/U另用于限速补值分组；MOVES road4/5用于统计运行系数。

### 2.4 逐边限速解析与缺失补值

每条有向边先读取适用的maxspeed:forward或maxspeed:backward，再读取通用maxspeed。适用方向依据原始way方向和本边方向确定。mph乘1.609344转成km/h；无单位数字按km/h解析。可解析多值采用最小值，保留原始字段和解析结果。

缺失、无效或不可解析限速，依次采用有效来源样本的道路长度加权中位数：

1. 同highway道路类。
2. 同H/M/U分组：H为motorway/trunk；M为其连接道及primary/secondary/tertiary等；U为residential、service、living_street、unclassified等。
3. 预定义parent class，如primary_link到primary、service到residential。
4. 本城CLE全部有效限速样本。

补值池覆盖本CLE共同道路图，包括城外包络道路。所有统计从补值前的有效样本计算。每边保存限速值、来源、补值层级及imputed标记。

### 2.5 有效速度与低流量基准

对州s、道路类r、日型d及小时h，速度档位代表值为v_b、VHT份额为p_{srdhb}、小时VMT权重为w_{srdh}。先算小时速度，再按VMT/VHT聚合：

$$
u_{srdh}=\sum_b p_{srdhb}v_b,\qquad
V_{srd}^{eff}=
\frac{\sum_{h\in H}w_{srdh}}{\sum_{h\in H}w_{srdh}/u_{srdh}}.
$$

参考聚合窗口H默认本地08:00–24:00，是已确认的可配置配送日参考时域；该时段端点属于本项目设计参数，不是官方规定或Amazon统计推导值。MOVES hourID9对应08–09，hourID24对应23–24。

低流量基准采用周末06:00–10:00，按VHT池化速度档位后取85分位。该低流量定义参照FHWA自由流速度估计思想[本节参考3]，在本项目中适配为州级VHT分箱估计，并非逐街实测自由流速度：

$$
a_b=\sum_{h\in L}\frac{w_{sr,\mathrm{weekend},h}}{u_{sr,\mathrm{weekend},h}}
\,p_{sr,\mathrm{weekend},h,b},\qquad
V_{sr}^{low}=Q_{0.85}\!\left(v_b;\frac{a_b}{\sum_j a_j}\right).
$$

同州、同道路类的工作日及周末使用同一个周末低流量分母，各自的有效速度形成对应运行系数。

### 2.6 逐边参考速度与时间

$$
\rho_{srd}=V_{srd}^{eff}/V_{sr}^{low},\qquad
v_{ed}^{ref}=\min(v_e^{limit},\rho_{srd}v_e^{limit}),\qquad
t_{ed}=3.6\,\ell_e/v_{ed}^{ref}.
$$

长度以米、速度以km/h、时间以秒保存。每州采用road4/5乘工作日/周末共四个系数。配置使用JSON原精度，下表为显示值：

| 州 | road4工作日 | road4周末 | road5工作日 | road5周末 |
|---|---:|---:|---:|---:|
| AZ | 0.760860 | 0.830100 | 0.495657 | 0.566484 |
| CA | 0.709587 | 0.804930 | 0.528933 | 0.580674 |
| FL | 0.720111 | 0.774017 | 0.511761 | 0.571466 |
| IL | 0.732897 | 0.782682 | 0.537175 | 0.526697 |
| NY | 0.597402 | 0.672079 | 0.510966 | 0.533801 |
| PA | 0.731445 | 0.775049 | 0.490852 | 0.513581 |
| TX | 0.737742 | 0.816223 | 0.572755 | 0.584258 |

同州同类共享相对系数，逐边限速保留不同道路的绝对速度尺度。该构造定义为州统计支持的静态参考profile；道路中的住宅、主干道和service子类使用同一参数合并假设。原始分类、限速和profile身份随输出保存。

### 2.7 共同原生边绑定与输出

配送路由图及用途接入图按原生有向边(u,v,key)合并成速度计算库存，每条边统一赋值并保存用途membership。部分边及进出路径按实际源边比例累计长度和时间。

几何connector保存距离d及时间换算系数3.6d，供运行场景指定connector速度后计算时间。输出包含原始/解析限速、赋值来源、补值标记、道路分类、profile_id、两种日型的rho、参考速度及通行时间。

### 2.8 参考文献与数据入口

1. [EPA 2023 NEI技术说明，第5.5.4节](https://gaftp.epa.gov/Air/nei/2023/doc/TSD/onroadmobile-agencymovesinputs.html)；[EPA MOVES5车型、道路及活动技术报告](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P101CUN7.TXT)。
2. [州profile、源文件和转换SQL注册表](../configs/us_nei2023_streetlight_state_profiles_v1.json)。
3. [FHWA-HOP-16-082，第5章Free Flow Speed](https://ops.fhwa.dot.gov/publications/fhwahop16082/ch5.htm)；[FHWA-HOP-21-058，第3章Measures Calculation](https://ops.fhwa.dot.gov/publications/fhwahop21058/chap3.htm)。
4. [Heni、Renaud和Coelho，CIRRELT-2018-14，第3节式(1)，研究技术报告](https://www.cirrelt.ca/documentstravail/cirrelt-2018-14.pdf)。
5. [FHWA，Sample Methodologies，Appendix：Free-flow speed estimation](https://www.fhwa.dot.gov/Environment/air_quality/conformity/research/sample_methodologies/emismeth07.cfm)。

### 2.9 代码落实与当前版本统计

复用[MOVES分布到州系数](../src/evrptw_cle/moves_speed.py)、[限速解析/补值](../src/evrptw_cle/speed.py)、[共同原生边绑定](../src/evrptw_cle/common_speed_binding.py)。州profile注册表及源SQL随代码提供；服务器以最新路由/接入图的原生边并集重新形成逐边速度库存。

每州保留16个速度档、两种道路类、两种日型和24小时维度。逐边统计区分原生限速observed与imputed，同时计算条数和长度占比。每次构造按[统计定义](stage1/STATISTICS.zh-CN.md)记录该次共同图的实际数量。


## 3. Customer location：building with residential units

### 3.1 建筑先行

主要几何使用 Microsoft Global 2026-08-13 索引快照。完整 OSM building 提供用途、可信整栋别名，以及复合 Microsoft 轮廓的子栋。双方唯一正面积相交、IoU≥0.5 时形成整栋别名；一个 MS 轮廓包含至少两栋有实质支持的 OSM 子栋时，以完整子栋替代父轮廓并保留父子 ID。

canonical polygon 的内部代表点决定服务域归属。候选分母包含范围内的明确住宅和用途未知建筑；明确非住宅、混合用途及用途未知且清楚 NSI=0 的对象分别排除/列示。零/正冲突、无 NSI、未补全不因结果缺失而从候选统计中消失。House/apartments/generic 只作邻居模型分组，所有地点最终都按建筑计数。

### 3.2 NSI 空间与户数归属

NSI 2026 提供普通 RES1/RES2/RES3 的正户数点、来源 footprint 标识、BID 框和面积。点包含、U 形院落支持、来源 bbox IoU≥0.25、面积对称比≤4、长/短轴归一化偏移共同形成候选。偏移阈值为 `0.2853432473555575`，来自固定 silver 参考的分层 Q99。[NSI 技术说明](https://www.hec.usace.army.mil/confluence/nsi/technicalreferences/2026/technical-documentation)

完整 OSM containment 参与候选约束。对合法候选做来源组—建筑的全局匹配：优先直接包含，再优先推断匹配，再按邻居/来源面积/高度差异和稳定 ID 解决并列。每条 NSI、真实 footprint 来源组和建筑最多选择一次。最终取一条 NSI `resunits`，不求和、不分摊小区总户数。全部未选来源保留账本。

### 3.3 建筑规模、参考池与 RMS 邻居

用 polygon 面积及独立 Microsoft/OSM 高度、OSM 层数建立宽松规模代理：楼层取 $\lceil H/3.048\rceil$ 与 $\lceil L\rceil$ 的较大值，再以总面积除以 10 m² 作为户数上界代理。两个参数可配置；一户来源支持保留；最终 U>1 必须规模可评估且通过。缺失高度可先补全代理，仍无法评估则记 `scale_unassessed_multiunit`，不计准入。NSI 自身由户数推导的楼层不参与独立筛查。

清楚的单条 NSI 住宅匹配形成固定参考池。按 2 km 空间块及相关来源组分隔 train/calibration/test，默认 80/10/10；参考不含 neighbor 估计。每个 city/type/可用垂直分支，用最多 2,000 个训练探针及地理最近 8 个独立参考拟合径向距离、log 面积、log 独立高度/层数差异的 RMS 标度。

三个可用特征组在拟合比较对上的平均平方贡献相等；x/y 合为一个空间组。仅有面积与位置时使用两组。候选相似度为标准化差异的均方根，最近 8 个参考以 $ (1+d)^{-2} $ 加权，log 户数中位数还原为正整数。原始户数与 NSI 户数档位不作为待预测特征。

相似度成本与 log 户数 Q90−Q10 分散度在留出校准参考上联合定界，默认保留 95% 参考支持，至少 8 个邻居。门槛按 city/type/branch 拟合，样本不足时按既定层级合并。95% 是支持目标。缺失高度可用同类型原始独立高度参考、area 分支距离的加权 P90 补充规模代理，保留估计标记；不将补高回流为原始预测特征。

### 3.4 准入与来源

有合法 NSI 代表则采用该条原户数；无代表、无原生冲突、邻居支持成立且规模允许时采用 neighbor 估计。原归属的全部来源若均为规模异常，不以邻居估计覆盖该异常。其余保留空值和原因。

每栋输出 `units_source=nsi|neighbor|unresolved`、`units_method=nsi_representative|neighbor_estimated`、`selected_nsi_fd_id`、原始/最终户数、独立/估计高度和支持/规模诊断。原始来源记录不覆盖。

当前 11 城固定候选 4,012,462 栋，其中 NSI 代表 3,194,252、邻居估计 728,489，总属性覆盖 3,922,741（97.7639%），未补全 89,721。这是道路接入前的候选建筑属性覆盖率。逐城表与未补全原因见[统计](stage1/STATISTICS.zh-CN.md)。

最终完整 canonical polygon 投影到共同道路。道路接入后可用数另计，接入未解决的建筑继续保留其属性和状态。公式、边界处理及完整选择顺序见[Customer 规格](stage1/CUSTOMER_METHOD.zh-CN.md)。

## 4. Depot location

### 4.1 输入与候选

输入为同一冻结OSM PBF的设施点/面及原始标签。具有配送设施功能支持的承运商/dispatch证据为A类；仓储/物流代理及普通office=logistics保留B类；泛工业区域和零售寄件柜台等C类只作背景，不作为默认depot。品牌、operator或泛工业用途本身不确认出发设施。

### 4.2 地点归并

先去除原生重复及有明确支持的同设施别名，再将**明确物流园区边界/成员**支持的设施整体归为一个depot，允许多个物流公司同园合并。泛landuse=industrial区域不整片合并，不以统一距离将邻近设施串联。完整园区范围、运营者和成员保留。

### 4.3 道路接入与输出

全部成员使用共同道路规则分别计算进出接入；组内有合格成员时稳定选择可用代表，其他成员仍保存。设施类别和名称不触发空间排除；原生禁行、方向、锁闭及障碍仍有效。

输出设施候选、canonical地点、成员、园区/别名证据及有向接入。代码复用[depot_candidates.py](../src/evrptw_cle/depot_candidates.py)、[depot_grouping.py](../src/evrptw_cle/depot_grouping.py)、[facilities.py](../src/evrptw_cle/facilities.py)。细节与伪代码见[设施实现规格](stage1/DEPOT_CHARGING.zh-CN.md)。

## 5. Charging station location及设备

### 5.1 AFDC坐标准入

AFDC station提供地点、坐标精度、状态及访问条件；equipment响应提供原生设备、接口和功率。GPS/Point以有效原生坐标独立准入；Address通过归档地址响应辅助核对，原生点与匹配辅助点相距≤250m为已有工程准入参数，不代表250m定位精度。OSM仅提供道路接入，不新增CS、补功率或扩大归并。

### 5.2 同址归并

按原生身份及“同坐标＋规范化完整地址”识别同址记录。同地址、不同坐标不直接合并；同坐标但身份/地址冲突隔离。归并后保留全部member、设备、端口、访问条件。默认库存为公共、运行状态E；设施类别mask不参与。

### 5.3 兼容配置与功率

Stage 1保存原生设备库存及具名参考profile。参考2025 Rivian Delivery700采用AC J1772、DC CCS1，不默认适配器；输入上限AC11kW、DC100kW。[厂商参考手册第6页](https://assets.ctfassets.net/2md5qhoeajym/5FQcJgfAOa4vDYu9rWwEYO/2fa75339d6e533532ba08bf395275015/RCV-QuickRef-v17.pdf)

先按源member身份/地址/坐标绑定设备，再独立枚举兼容AC/DC。设备坐标绑定容差1m是来源一致性参数。同址多个端口功率不相加。

原生reported功率有效则保留；缺失时采用同一冻结AFDC设备库存、按最新资格筛选后计算的**同模式端口数加权全国中位数**。记录原始值、插补标记及车辆输入上限，参考功率取assigned与车辆上限较小者。中位数随本次冻结设备库存及参数版本记录。

地点参考配置优先有效reported，再在该层选vehicle-capped power最大配置，最后稳定ID破并列；保留全部其他配置。Stage 1不乘0.9降额，不确定实例充电策略。

### 5.4 接入、输出与代码落实

在共同路网上按charging用途计算有向接入；静态地点资格、road-accessible与参考兼容性分别保存。输出location/member/configuration及接入表。

复用[charging_component.py](../src/evrptw_cle/charging_component.py)、[charging_parameters.py](../src/evrptw_cle/charging_parameters.py)、[charging_road_access.py](../src/evrptw_cle/charging_road_access.py)；绑定共同道路图，参数profile外置。算法、缺失功率中位数计算与函数接口见[设施实现规格](stage1/DEPOT_CHARGING.zh-CN.md)。

## CLE组装与Stage 2接口

三类地点统一以(city, location_id)标识。保留来源实体ID、canonical几何、geo_assignment_point、原生成员/NSI owner、道路anchor、必要的进出gateway及其实际路径。

给定距离或时间权重w：

$$
C_{ij}^{(w)}
=C_i^{conn,out}(w)+C_i^{terminal,out}(w)
+d_{G^{route}}^{(w)}(g_i^{out},g_j^{in})
+C_j^{terminal,in}(w)+C_j^{conn,in}(w).
$$

connector、真实terminal及核心道路分别计价，全部道路时间引用共同原生边速度。普通private在路由核心中参与路径。Stage 2读取固定CLE后再选择depot/customer/CS，生成需求及实例矩阵。

组装和读取器保留建筑级 customer、可空户数及字段来源，并绑定共同有向边。完整接口和构造顺序见[运行说明](stage1/SERVER_IMPLEMENTATION.zh-CN.md)。当前建筑属性统计已完成 11 城离线重算；完整 CLE 的道路后可用数以该次组件构造和组装 manifest 为准。
