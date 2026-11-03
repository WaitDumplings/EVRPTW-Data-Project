# Stage 1 细节附录：4. Depot location 与 5. Charging station location

本文补充[Stage 1主规格](../STAGE1_SPEC.zh-CN.md)中的两类设施流程。Stage 1建立城市中可供后续选择的地点及其道路接入、来源属性；实例如何选depot、激活多少CS及访问哪些站点属于Stage 2。

本文按来源、分类、地点身份、设备及共同道路接入介绍当前设施构造。明确物流园整体归并，泛工业区域保留设施级身份；普通 private 按共同配送路网假设通行。来源与访问资格、道路可达性分别保存。

## 4. Depot location

### 4.1 第一步：从共同OSM快照读取设施与原始证据

Depot和道路采用同一Geofabrik OSM PBF，replication时间为**2026-10-25T21:21:21Z**。PBF中的node、way、relation及标签是原始资料；提取结果保存为WGS84几何的GeoParquet，包括Point、Polygon或MultiPolygon，以及原生OSM身份和用途、名称、运营者、地址等标签。

提取器保留物流设施以及可能为归组提供上下文的对象，主要信号包括building=warehouse、industrial=warehouse/logistics/distribution、landuse=warehouse/logistics/depot/industrial、office=logistics、amenity=post_depot及明确depot/logistics标签。这里提取的泛工业对象只是待分类背景，不能直接视为depot。

每城extraction.json保存PBF URL/SHA、快照时间、边界hash、原始对象数量及提取器身份。快照日期表示获取的OSM状态，并不证明所有设施在2026年经过现场调查；对象编辑日期也不等同于设施启用或观测日期。

来源入口：`data/frozen_20261027/sources/depot/<city>/extraction.json`及同目录`raw_osm_depot_evidence.parquet`。分类代码为[depot_candidates.py](../../src/evrptw_cle/depot_candidates.py)。

### 4.2 第二步：区分潜在设施与背景对象

从原始属性与几何产生三个证据类别：

| 类别 | 依据 | 地点资格 |
| --- | --- | --- |
| A | 承运商或dispatch设施，并有仓储、配送功能名称或明确设施标签支持 | 潜在depot |
| B | warehouse/logistics/depot代理，或普通office=logistics | 潜在depot |
| C | 只有泛工业背景、零售寄件柜台，或明确非地面物流的office等 | 不作为候选depot |

品牌名本身不足以确认实际配送设施。例如，UPS/FedEx名称还需设施功能支持；零售寄件柜台不按配送出发设施处理。普通logistics office保留为B类，表示有物流用途证据的代理地点，不声称已核实该办公室每日有配送车辆出发。

设施面积作为连续属性保留，不设面积准入门槛。A/B用于说明证据来源，不设置Stage 2抽样优先级。

原始几何和所属城市以service boundary确定；多边形可取内部代表点作地点与边界归属。这里的代表点是空间表示，不声称真实装卸门口位置。几何面积、距离在城市米制投影下计算，输出经纬度仍为WGS84。

### 4.3 第三步：设施别名去重，再按明确物流园区归并

这一层先去除原生重复及有证据的同设施别名，再将明确园区内的设施整体归并。地点单位是明确物流园区或独立设施，而不是公司。一处原生OSM对象、边界或明确成员关系支持的物流园区，其A/B设施整体归为一个location，允许多个物流公司和多个仓库。原生园区身份、边界、设施成员与各运营者一并保留。

例如，明确园区内有三家公司、六个仓库，Stage 1形成一个depot location；同一片泛工业用地中的六个无明确园区关系的仓库，不因处于工业区而整体合成一个depot。

前一步的重复POI、设施面和地址表示使用以下全国统一设施别名规则；这些别名参数不决定园区边界：

| 证据 | 归并方式 |
| --- | --- |
| 原生共同设施身份或明确成员关系 | 保留共同身份及成员 |
| Point落在同一设施Polygon中 | 作为同一设施的点/面表示处理 |
| 两个设施Polygon重叠 | 交叠覆盖较小面积至少80%时视为别名 |
| Point坐标别名 | 相距≤1m，配合新增组跨度保护 |
| 完整门牌号与街道相同 | 代表点≤100m，配合新增组跨度保护，避免链式连接远处设施 |
| 泛工业用地 | 只保存背景关联，不据此并组 |

1m、80%和100m是设施别名识别的可配置工程参数，用于坐标别名、重复轮廓与完整地址表示，不是从现实物流园尺度统计估计出的值。它们不决定园区边界或园区大小；明确园区按原生身份、范围及成员归并。派生显示名（如Unnamed warehouse）不构成共享名称证据；地址匹配保留原始结构化地址。

设施别名和明确园区身份由 [depot_grouping.py](../../src/evrptw_cle/depot_grouping.py) 与 depot 构造层处理，保留别名、园区成员及其来源。

### 4.4 第四步：把各设施成员接入共同有向路网

归并前的设施成员与接入证据保留。对每个成员，以源代表点及原设施几何作为接入搜索依据，使用Stage 1共同路网的局部候选道路检查，保存投影点、connector、原生有向边、入口/出口及公共图连接信息。路网中的oneway等方向和适用的原生禁行标签继续生效。

普通 private 道路参与配送穿行及 shortcut；源标签明确禁止相应车辆/配送、单行方向、锁闭及物理障碍按共同规则处理。设施类别和名称本身不生成空间排除，限制作用于原生对象及其真实几何。

地点是否在来源上成立，与是否完成道路接入分别保存。无法确认局部连接或有向往返接入的成员保留状态和原因，不按1km阈值直接删除，也不逐点手工移动坐标。

接入代码可复用[facilities.py](../../src/evrptw_cle/facilities.py)的build_depot_layer、[local_road_access.py](../../src/evrptw_cle/local_road_access.py)及[terminal_access.py](../../src/evrptw_cle/terminal_access.py)；三类地点共享主规格的 private、原生禁行及方向规则。

### 4.5 第五步：形成canonical depot与保留的成员表

每个facility_group_id形成一个canonical地点。组内有合格接入成员时，先选可用成员，再按candidate_id稳定排序取得代表；不按A类优先。canonical行引用该代表的接入信息，所有源成员、标签、归并理由、运营者及接入行继续保留。

因此，成员表的行数与地点数量不同。地点总数按facility_group_id去重；可用地点数是在depot_candidate_eligible=True成员中对facility_group_id去重。CLE中登记潜在出发地点，并不在这一阶段选定某个实例的depot。

实现与输出入口：[cli.py](../../src/evrptw_cle/cli.py)的depot流程、[cle_assembly.py](../../src/evrptw_cle/cle_assembly.py)、`depots/depots.parquet`及`depot_component_manifest.json`。

## 5. Charging station location

### 5.1 第一步：读取AFDC站点库存和设备库存

充电地点、原生坐标、接口和功率以AFDC站点及equipment为主资料。OSM只负责共同路网接入；不以OSM充电POI新增站点、补功率、扩大归并范围或覆盖AFDC原生坐标。

| 输入 | 格式与内容 | 获取时间/范围 |
| --- | --- | --- |
| AFDC station API | CSV；ID、名称、Street Address/City/State/ZIP、Latitude/Longitude、Geocode Status、公开访问、状态、接口/端口摘要与开放条件等 | 2026-10-26T23:28:25.956900Z；US、ELEC、public、status E |
| AFDC charging units API | CSV；站点身份及设备/connector计数、J1772/CCS等模式功率、访问等来源属性 | 2026-10-27T00:03:57.499877Z，即洛杉矶本地10/26；独立归档响应 |
| Census地址辅助 | CSV；AFDC ID、回显原输入地址、match状态及辅助经纬度 | 2026-10-26T23:47:41.254276Z；Public_AR_Current地址范围 |

station和equipment是分别归档的滚动响应，不宣称同一瞬间的快照。快照时间与记录的UpdatedAt、DateLastConfirmed、OpenDate分开保存；这些字段也不能统一解释为功率实际测量年份。每份原始响应保留来源URL、查询条件、获取时间与SHA。

来源元数据见`data/frozen_20261027/sources/afdc/`的三个manifest：`afdc_us_public_available_electric_20261026.manifest.json`、`afdc_us_public_available_charging_units.manifest.json`与`afdc_address_only_census_20261026.manifest.json`。[AFDC设备接口说明](https://developer.nlr.gov/docs/transportation/alt-fuel-stations-v1/ev-charging-units/)用于解释设备接口字段。

### 5.2 第二步：按坐标来源准入，保留AFDC原点

CSV中的经纬度转换为WGS84 Point，并按同一城市service boundary保留城内原生点。原经纬度、坐标来源层级、辅助验证信息分别保存。

| Geocode Status | 全国统一规则 |
| --- | --- |
| GPS或200-9 Point | 原始经纬度为有限有效值且在经纬度范围内，独立准入 |
| 200-8 Address | Census返回match、回显输入与当前AFDC原地址一致，且原生点到辅助点大圆距离≤250m时准入 |
| 缺失、非法值或其他无法佐证的来源 | 保存未确认状态及原因，不强制替换成辅助点 |

Address通过验证后仍使用AFDC原生点。250m是地址辅助一致性参数，不是GPS精度承诺，也不是道路投影删除阈值。Census提供地址范围插值，不能当作观察到的设备位置或车辆入口。同地址OSM对象即使存在，也不用于替换原生坐标。坐标流程不使用逐记录manual overrides。

当前[charger_resolution.py](../../src/evrptw_cle/charger_resolution.py)实现该规则，组件构建调用不传入OSM充电POI。

### 5.3 第三步：统一识别同址记录、访问资格与冲突

地点身份采用**相同原生经纬度＋完整归一化street/state/ZIP5**。地址归一化处理大小写、标点、街道方向与后缀，完整键需有门牌、街道、州和ZIP5。这里是相同坐标的来源别名识别，不进行任意距离聚类。

- 同坐标、同完整地址：合成一个location，保留所有AFDC成员ID。
- 同完整地址、不同原生坐标：保持不同location。
- 同坐标碰撞且地址不同，或碰撞中有缺失地址：隔离冲突成员，不擅自判断哪条坐标正确。
- 地址不完整但没有同坐标碰撞的单条记录：保留独立身份。

来源status E和public是库存准入条件；若访问详情或原文字明确显示仅员工、住户、政府或fleet等内部使用，或显示已移除、永久关闭等与结构化状态冲突，按统一原因保留为不合格/待核。该处理保存来源访问/有效性判断。开放时间、访问限制、网络和收费按原值保存，不把public推定为24小时开放。

一个独立合格成员可以支持同址地点；冲突或不合格成员不为该地点提供设备资格。归并后保留members表及equipment配置，不跨来源别名相加端口或功率；location_total_ports不由别名行数推算。

实现：[charger_locations.py](../../src/evrptw_cle/charger_locations.py)的assign_charging_locations及collapse_charging_locations。

### 5.4 第四步：分别保存潜在地点、道路接入和参考车辆兼容

AFDC原点使用与customer/depot同一物理来源的共同有向路网，按charging用途检查接入。局部投影、connector、道路方向、连接路径及未确认原因随地点保存；原点不会因投影而被覆盖。250m道路距离标记是QA信息，不作自动删除条件。

三个资格分别表示：

| 字段 | 含义 |
| --- | --- |
| potential_location_eligible | 坐标、来源身份、公开访问等静态地点规则成立 |
| road_accessible | 已确认局部道路接入 |
| reference_vehicle_compatible | 同址存在合格的原生兼容设备配置 |
| charging_parameter_eligible | 上述三者同时成立 |

道路与兼容在各自对象上计算，不必是逐步缩小的漏斗。某地点可以道路可达但只提供不兼容接口，也可以有兼容设备但暂时无法确认接入；完整潜在库存与配置均保留。

普通 private 按共同道路方案通行；原生禁令及站点自身公开/非公开访问条件分别保存和求值。

实现：[charging_component.py](../../src/evrptw_cle/charging_component.py)的build_charging_layers、[charging_road_access.py](../../src/evrptw_cle/charging_road_access.py)。

### 5.5 第五步：在同址设备中枚举兼容配置

Stage 1保存原生设备库存，并给出一个命名参考profile。当前参考是**2025 Rivian Commercial Van Delivery700**：AC J1772最大11kW，DC CCS1/J1772COMBO最大100kW，不默认适配器。依据为[厂商2025 QuickRef v17第6页](https://assets.ctfassets.net/2md5qhoeajym/5FQcJgfAOa4vDYu9rWwEYO/2fa75339d6e533532ba08bf395275015/RCV-QuickRef-v17.pdf)。车型文件于2026-10-27T00:02:24.249136Z归档；文件SHA、页码和车型属性保存在[charging_parameter_contract_v1.json](../../configs/charging_parameter_contract_v1.json)。

设备先通过源成员资格及站点ID、地址、坐标绑定，再按接口枚举AC/DC。坐标绑定沿用1m来源一致性参数，不表示道路接入距离。未知车辆净空保留不确定性；该van GVWR为9500lb，Class2/LD本身不构成接口不兼容。

AC与DC独立枚举，因此一个地点虽然DC只有CHAdeMO，仍可能通过同址J1772 AC成为参考兼容地点。功率缺失不会被用来推断接口是否兼容。

### 5.6 第六步：赋功率、施加参考上限并选一个配置

每个合格兼容配置保存reported_power_kw与功率质量标记。非有限或非正功率按未知；J1772 L2报告值>19.2kW时保留原值与QA，赋值按未知处理。

对已经确认兼容的配置c，按模式m赋值并施加参考车辆输入上限：

$$
P_c^{assigned}=\begin{cases}
P_c^{reported},&\text{原生功率有效},\\
m_m,&\text{兼容已确认且原生功率未知},
\end{cases}
\qquad P_c^{ref}=\min(P_c^{assigned},P_{max,m}^{ref}).
$$

全国模式中位数从本次绑定的冻结 AFDC 库存构造。AC与DC各计算一次，所有城市共用相应模式值：

1. 读取全国station、equipment及已归档Census辅助，不先裁入某个城市；按本节US/ELEC/public/status E、坐标准入、同坐标冲突和访问资格产生合格source members，不应用设施类别或派生缓冲mask。缺失地址辅助的Address记录按同一准入规则保持未确认，不人为补坐标。
2. 按AFDC ID将设备绑定到合格member，使用同样的地址、坐标与访问一致性规则。对AC J1772和DC CCS1分别选择对应connector count为正的源设备观测。
3. 仅用原生有效reported功率形成校准样本；非正/非有限值及本节J1772 L2功率QA标记不进入中位数样本。功率缺失值不先插补再参加统计。
4. 以该模式的源connector count作权重，对reported功率升序排列，取累计权重首次达到总权重一半的功率。即

$$
m_m=\inf\left\{p:\sum_{c:P_c^{reported}\le p}n_{c,m}\ge
\tfrac12\sum_c n_{c,m}\right\}.
$$

这里的权重是来源设备观测中的该模式端口数，不是跨AFDC别名推算的地点总容量；同一来源设备/connector的重复提取不重复计权。统计发生在车辆功率截断之前，也不依赖某城市道路接入是否成功。生成模式中位数、有效样本与权重数量、资格/QA规则、输入SHA及构造版本，供各城共同引用。[us_national_charging_power_medians_v1.json](../../configs/us_national_charging_power_medians_v1.json)记录模式统计、来源及构造参数；每次运行引用其实际版本。

同一地点的所选配置按以下优先级确定：

1. 优先有效reported功率配置。
2. 在该优先层内选择vehicle-capped power最大的配置。
3. 按AFDC ID、设备来源行稳定打破并列。

因此，“采用最快兼容功率”的完整含义包含来源资格、reported优先和车辆上限。例如，同址有效DC150与350kW都被参考van限制为100kW，并以稳定规则选其一，而不相加为500kW；若只有兼容AC已报告7kW与功率缺失DC，reported优先策略会选择7kW配置，DC插补候选仍留在配置表。这里不能简写为在全部assigned_power中无条件取最大。

保存reported、assigned、vehicle_cap及capped_pre_derating功率，连同selected_configuration_id和source标记。**Stage 1不乘0.9**；它也不确定真实车队、电池、续航、访问顺序或full-charging策略。若后续Stage 2使用降额，应由场景配置一次施加。

设备赋值入口为 [charging_parameters.py](../../src/evrptw_cle/charging_parameters.py)的select_reference_configurations；接口、11/100 kW 参考上限和功率 QA 从统一运行配置读取。原生设备功率与参考profile分开保存。

### 5.7 第七步：形成地点表、成员表和设备配置表

`chargers.parquet`每个charging_location_id一行，保留三类资格、所选配置、原生点和道路接入；`charging_members.parquet`保存原始AFDC站点成员；`charging_configurations.parquet`保存可追溯设备配置。换参考profile时仍可读取原生库存，不必将此前不兼容地点解释为不存在充电设施。

## 6. 服务器实现的输入、顺序与输出

### 6.1 输入契约

| 输入 | 必需内容 |
| --- | --- |
| 城市及边界 | city ID、WGS84 service boundary、本城米制CRS与归属规则 |
| 共同OSM来源 | 同快照PBF/提取，完整设施与园区几何、原生身份/成员/标签及来源SHA |
| 共同路网 | 按主规格形成的有向routing与用途接入图、reference SCC、原生边及速度身份 |
| AFDC资料 | frozen station CSV、equipment CSV、与原输入绑定的Census辅助及各自manifest |
| 运行配置 | 设施别名参数、Address辅助参数、设备绑定参数、参考接口及上限、功率QA规则 |
| 新算模式统计 | 根据5.6节从全国冻结AFDC计算的AC/DC中位数及其来源manifest |

所有城市使用同一规则与运行配置结构，城市输入只决定实际地理对象、CRS和道路内容。构造按以下顺序保存中间证据与结果。

### 6.2 Depot伪代码

```text
objects := read_osm_logistics_evidence(common_snapshot, city, service_boundary)
objects := preserve_native_ids_geometries_tags_and_members(objects)
candidates, context := classify_A_B_C(objects)  # A/B候选；C为背景

facility_groups := deduplicate_native_objects_and_facility_aliases(
    candidates,
    point_alias_m = config.point_alias_m,
    overlap_fraction = config.overlap_fraction,
    complete_address_m = config.complete_address_m,
    bounded_alias_group_span = true
)
park_membership := resolve_explicit_logistics_park_identity(objects, context)
groups := merge_facility_groups_with_explicit_park_membership(
    facility_groups, park_membership
)  # 不由generic industrial context产生并组

members := anchor_each_source_member(
    groups.members, common_directed_road_context, purpose = delivery
)  # 普通private可穿行；原生禁令、方向及实际障碍继续生效
canonical := choose_member_per_group(
    members, order = [available DESC, candidate_id ASC]
)
write_depot_locations_members_membership_and_access(canonical, members, groups)
```

A/B分类不调用设施敏感类别/名称排除分支；明确访问禁令仍由适用用途与原生几何处理。园区身份与原始设施别名分别存储，后续能追溯一个location为何包含这些设施。

### 6.3 Charging伪代码

```text
national_members := resolve_afdc_native_coordinates(stations, census_aux)
national_members := apply_public_operating_source_and_access_qualification(
    national_members
)  # 不作设施类别mask
national_locations, national_members := assign_same_coordinate_address_identity(
    national_members
)
national_configs := bind_equipment_and_enumerate_native_compatible_modes(
    equipment, national_members, reference_profile
)
mode_medians := port_weighted_reported_power_medians(
    national_configs, power_QA = config.power_QA
)  # 先统计；不使用imputed值，不作车辆截断

members := native_points_assigned_to_city(national_members, service_boundary)
locations := collapse_members_preserving_all_sources_and_access(members)
configs := compatible_configurations_for_city(national_configs, members)
configs.assigned := reported_if_valid_else_mode_median(configs, mode_medians)
configs.capped_pre_derating := min(configs.assigned, reference_mode_cap)
selected := choose_one_configuration_per_location(
    configs,
    order = [reported DESC, capped_pre_derating DESC, AFDC_ID ASC, unit_row ASC]
)

locations := anchor_afdc_native_points(
    locations, common_directed_road_context, purpose = charging
)
locations.charging_parameter_eligible := (
    locations.potential_location_eligible
    AND locations.road_accessible
    AND locations.reference_vehicle_compatible
)
write_charging_locations_members_configurations_and_access(
    locations, members, configs, selected
)  # Stage1不乘0.9
```

national_configs在统计阶段保留有效、缺失功率状态；只有有效reported值进入中位数。city阶段再为缺失配置赋值，避免循环定义。设备配置来自合格member；道路资格不反向改变原生设备属性或全国统计样本。

### 6.4 输出契约

| 输出层 | 主要字段与含义 |
| --- | --- |
| Depot canonical | location/group ID、内部归属代表点、明确园区身份、选中成员、地点与道路资格 |
| Depot members | 原生OSM ID/标签/完整几何、A/B证据、设施别名/园区成员、归组理由、各自接入 |
| Charging locations | charging_location_id、AFDC原点、归属、静态/道路/参考兼容三类资格、选中配置及接入 |
| Charging members | AFDC ID、原经纬度、Geocode Status、地址辅助、地点身份与冲突、访问/时段/网络信息 |
| Charging configurations | 原设备来源行、member ID、模式/connector/端口、reported/QA/assigned/cap/capped-pre-derating、所选标记 |
| Access | anchor、connector、原生有向边、进出gateway、必要terminal路径及相同道路身份下的成本 |
| Manifest/config | 输入SHA、来源/规则/实现版本、米制CRS、参数、全国模式中位数来源及输出身份 |

成员行数、地点数、兼容配置数和端口权重分别记录，不能互相替代。地点资格、道路资格和参考兼容性分别保存；后续服务器测量结果再形成对应数据表。

## 7. 实现入口与输出绑定

| 职责 | 代码 |
|---|---|
| Depot 原生候选、用途证据 | `depot_candidates.py` |
| 设施别名与明确物流园身份 | `depot_grouping.py` |
| Depot 成员接入和地点代表 | `facilities.py`、`cle_assembly.py` |
| AFDC 原生坐标与辅助验证 | `charger_resolution.py` |
| Charging 同址身份、成员和冲突 | `charger_locations.py` |
| 原生兼容设备、功率来源和参考配置 | `charging_parameters.py` |
| Charging 地点与共同道路接入 | `charging_component.py`、`charging_road_access.py` |
| 公共路由及用途接入 | `local_road_access.py`、`delivery_access.py`、`terminal_access.py` |

文件位于 `src/evrptw_cle/`。`evrptw-cle stage1 components ... --component all` 按共同配置构造各层并组装。每次 manifest 记录输入 SHA、规则、原生边和所选设备；设施成员数、canonical 地点数和道路可用数分别统计。
