from pathlib import Path
import json,csv,shutil,hashlib
P=Path(__file__).resolve().parent/'results';D=Path.home()/'Desktop';a=json.load(open(P/'summary.json'));runs={}
for p in sorted((P/'runs').glob('*.json')):runs[p.stem]=json.load(open(p))
assert len(runs)==20
fmt=lambda x:'—' if x is None else f'{x:.3f}'
ftime=lambda x:'未达到' if x is None else f'{x:.3f}'
def st(x):return f"{x['mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
def vec(x):return '['+', '.join(fmt(v) for v in x)+']'
label={'overall':'整体','force/low':'0–15N','force/high':'15–45N','force/0_2':'0–2N（不含2）','force/2_5':'2–5N','force/5_15':'5–15N','force/15_30':'15–30N','force/30_45':'30–45N','settled_hold':'固定等待后的保持段','continuous_dynamic':'连续动态Sweep/Full3D','other_transition_or_startup':'其他过渡/启动','post_reset_first_observation':'重置后首个观测','not_post_reset_first_observation':'非重置首观测'}
def gl(g):
 if g in label:return label[g]
 return g.replace('/low',' / 0–15N').replace('/high',' / 15–45N')
# Auditable flat companion tables, preserving each seed, both models and all metrics.
with open(P/'per_seed_metrics.csv','w',newline='') as f:
 w=csv.writer(f);w.writerow(['model','seed','group','n','angle_n','mag_mae_N','mag_rmse_N','fx_mae_N','fy_mae_N','fz_mae_N','fx_rmse_N','fy_rmse_N','fz_rmse_N','angle_mean_deg','angle_p95_deg','relative_mag_mae_pct','relative_mag_rms_pct','relative_vector_rms_pct','vector_nrmse_pct','absolute_point_pass'])
 for key,r in runs.items():
  for g,v in r['groups'].items():
   if not v['n']:continue
   w.writerow([r['name'],r['seed'],g,v['n'],v['angle_n'],v['mag_mae'],v['mag_rmse'],*v['component_mae'],*v['component_rmse'],v['angle_mean'],v['angle_p95'],v['relative_mag_mae_pct'],v['relative_mag_rms_pct'],v['relative_vector_rms_pct'],v['vector_nrmse_pct'],v['absolute_pass']])
with open(P/'transient_events.csv','w',newline='') as f:
 w=csv.writer(f);w.writerow(['model','seed','event','env','reference_force_N','peak_vector100ms_N','peak_magnitude100ms_N','peak_vector1s_N','peak_magnitude1s_N','peak_vector100ms_reference_pct','integral_vector2s_Ns','duration_above1p5N_s','longest_above1p5N_s','peak_angle100ms_deg','angle_above10deg_total_s','physical_start_s','physical_confirm_s','reset_within2s','recovered_without_reset'])
 for key,r in runs.items():
  for ev,rr in r['transients'].items():
   v=rr['per_environment']
   for i,e in enumerate(v['env']):w.writerow([r['name'],r['seed'],ev,e,*[v[k][i] for k in ['reference_force_N','peak_vector_100ms_N','peak_mag_100ms_N','peak_vector_1s_N','peak_mag_1s_N','peak_vector_100ms_reference_pct','integral_vector_error_2s_Ns','exceed_vector_1p5N_total_s','exceed_vector_1p5N_longest_s','peak_angle_100ms_deg','exceed_angle_10deg_total_s','physical_recovery_start_s','physical_confirmation_s','any_reset_within2s','recovered_without_reset_before_confirmation']]])
S=a['summary']['selected'];T=a['transient_summary']['selected'];terms=a['termination_statistics'];dec=a['decision']
allpoint=dec['absolute_point_all_primary_all_seeds_pass'];robust=dec['absolute_simultaneous_all_primary_supported_pass']
lines=['# 力估计器新版重复验收报告（acceptance_v2）','',
 '**最终应用验收：待确定，不能宣布正式合格。用户已明确“允许误差与恢复时间预算未确定”；本轮没有为通过成绩自行放宽门槛。**','',
 f"冻结候选为第6320轮权重（训练在6410轮停止），SHA256 `{a['lock']['model_sha256']['selected']}`。对照为第240轮。采用10个全新种子30101–30110，每个种子分别运行两模型，共20条完整MuJoCo闭环轨迹。每条112环境×6000步，五场景，0–45N分层覆盖，共12880000个有效帧；帧数不当作独立重复数。本轮未训练、未选模、未替换默认部署。",'',
 f"**原有绝对门槛的逐种子全部主分组通过：{'是' if allpoint else '否'}；同时考虑100项主要检查的统计保护后，所有主分组得到通过支持：{'是' if robust else '否'}。** 这两个结论都不替代尚未确定的应用误差/响应预算。",'',
 '|候选汇总范围|大小RMSE N|大小MAE N|方向P95 °|方向平均 °|Fz RMSE N|原门槛通过种子/10|', '|---|---|---|---|---|---|---|']
for g in ['overall','force/low','force/high']:lines.append(f"|{gl(g)}|{st(S[g]['mag_rmse'])}|{st(S[g]['mag_mae'])}|{st(S[g]['angle_p95'])}|{st(S[g]['angle_mean'])}|{st(S[g]['fz_rmse'])}|{sum(runs[f'selected_{seed}']['groups'][g]['absolute_pass'] for seed in range(30101,30111))}|")
point_bad=[f"{gl(g)}（{b['point_pass_seeds']}/10通过）" for g,b in a['primary_simultaneous_bounds'].items() if b['point_pass_seeds']<10]
supported_bad=[gl(g) for g,b in a['primary_simultaneous_bounds'].items() if b['evidence']=='supported_fail_under_t_assumptions']
lines+=['','逐种子出现未通过的主分组：'+('；'.join(point_bad) if point_bad else '无')+'。','',
 '在同时区间与t假设下支持未达标的主分组：'+('、'.join(supported_bad) if supported_bad else '无；其他分组仍需区分支持通过与证据不足')+'。','']
lines+=['','所有“均值 [区间]”均为10个种子各自指标的等权均值及近似95% Student-t置信区间。尤其“方向P95均值”是10个种子P95的均值，不是混合所有帧后的P95；区间也不是单帧误差覆盖区间。独立单位是种子，逐帧数据相关。10次是固定预算的工程重复扩展，不是经功效计算保证充分的认证样本量。','',
 '## 一、三项改进如何落实','',
 '1. **绝对与相对误差并列。** 旧门槛保持：各分量MAE≤1N、RMSE≤1.5N；大小MAE/RMSE≤1.5N；方向平均≤5°、P95≤10°，方向指标仅在真值≥2N时考核；0<F<2N的方向数学上存在，但本轮不作角度达标判定，零力方向不定义。相对误差仅在真值≥2N时计算；不对接近零的单轴分量相除。低于2N用绝对误差，真正零力另报误报比例。5/10/20%仅作候选预算敏感性对照，没有作为新正式门槛。','',
 '2. **保持段、连续动态、切换瞬态分开。** 全阶段成绩保持完整；另报固定等待后的保持段，Sweep/Full3D仍为连续动态。瞬态包含前100ms和1s峰值、2s内累计误差、超限总时长/最长持续时长及恢复时间；不把全阶段均值代替瞬态验收，不删除尖峰。允许恢复时间未确定，所以不擅自宣布100ms必须或可以通过。','',
 '3. **重复验证和边界判定。** 10个新种子逐对比较，5对候选先跑、5对对照先跑，顺序在采集前冻结。每对同一GPU、同一施力计划；跨种子给出区间。原5场景×2力段×10项绝对指标共100项主要检查采用Bonferroni同时95% t界限，减少多项检查误判。若界限跨越门槛，标记边界/证据不足，而不是强行通过或把微小超限称为显著退化。小样本t假设与偏态风险仍存在；近似t区间可能超出非负或百分比的物理边界，不应将区间端点解释为可发生的负误差；此法并非完整GUM不确定度评定或ISO认证。','',
 '## 二、相对误差定义与力段结果','',
 '令真值向量为y、预测为p，F=||y||、P=||p||、v=||p−y||。仅对F≥2N计算：大小相对MAE=mean(|P−F|/F)×100%；大小相对RMS=sqrt(mean(((P−F)/F)²))×100%；向量相对RMS=sqrt(mean((v/F)²))×100%；向量NRMSE=sqrt(mean(v²))/sqrt(mean(F²))×100%。这些量不同，向量误差同时反映大小和方向。分组成员的力大小仅四舍五入至0.001N处理数值边界，误差本身不舍入；5/15/30N边界归较低组。','',
 '|力段|大小RMSE N|大小相对RMS %|向量相对RMS %|向量NRMSE %|方向P95 °|', '|---|---|---|---|---|---|']
for g in ['force/0_2','force/2_5','force/5_15','force/low','force/high','force/15_30','force/30_45']:
 r=S[g];lines.append('|'+gl(g)+'|'+'|'.join(st(r[k]) if k in r else '未考核（<2N）' for k in ['mag_rmse','relative_mag_rms_pct','relative_vector_rms_pct','vector_nrmse_pct','angle_p95'])+'|')
lines+=['','### 未批准的相对预算对照（不等于验收合格线）','','|力段|指标|≤5%的种子/10|≤10%的种子/10|≤20%的种子/10|','|---|---|---:|---:|---:|']
for g,rs in a['relative_budget_sensitivity'].items():
 for k,bs in rs.items():lines.append(f"|{gl(g)}|{k}|"+'|'.join(str(bs[str(b)]['point_pass_seeds']) for b in [5,10,20])+'|')
lines+=['','### 零力与误报','','真值模长≤0.0001N为零力。下面分别看全阶段零力与固定等待后保持段的零力，避免将释放瞬态误报混同于保持段噪声。误报阈值只是诊断水平，不是新增的每帧合格线。','','|零力样本来源|预测模长P95 N|>1N比例 %|>1.5N比例 %|>2N比例 %|>5N比例 %|','|---|---|---|---|---|---|']
for g in ['overall','settled_hold']:
 r=S[g];lines.append(f"|{gl(g)}|"+'|'.join(st(r[k]) if k in r else '无样本' for k in ['zero_pred_p95_N','zero_false_positive_above_1_pct','zero_false_positive_above_1.5_pct','zero_false_positive_above_2_pct','zero_false_positive_above_5_pct'])+'|')
lines+=['','## 三、完整场景、保持段和连续动态结果','','|场景/分组|大小RMSE N|大小MAE N|方向平均 °|方向P95 °|Fx/Fy/Fz RMSE均值 N|逐种子原门槛通过/10|','|---|---|---|---|---|---|---:|']
for g,r in S.items():
 if 'mag_rmse' not in r:continue
 passed=sum(runs[f'selected_{seed}']['groups'][g].get('absolute_pass',False) for seed in range(30101,30111))
 lines.append(f"|{gl(g)}|{st(r['mag_rmse'])}|{st(r['mag_mae'])}|{st(r['angle_mean']) if 'angle_mean' in r else '不定义'}|{st(r['angle_p95']) if 'angle_p95' in r else '不定义'}|"+vec([r[k]['mean'] for k in ['fx_rmse','fy_rmse','fz_rmse']])+f'|{passed}|')
lines+=['','保持段诊断：Ramp达到目标后再等待1s（阶段内≥351）；Release加载后等待1s至释放前（101–400）以及释放后等待1s（≥451）；Reverse同理为101–600和≥651。此分类不移除正式全阶段成绩中的过渡帧。“其他过渡/启动”与保持段、连续动态一起完整覆盖有效帧。各阶段前50步为原有零力热身，不计入旧口径精度。重置后首观测单列诊断但继续保留在所有正式全阶段成绩中。这用于检查历史缓存重置时的信息不足，不能通过删除这些帧使总成绩通过。','',
 '### 100项主要检查的同时区间判定','','|主分组|逐种子通过/10|同时界限判定|超过门槛的均值下界项|','|---|---:|---|---|']
for g,b in a['primary_simultaneous_bounds'].items():
 bad=[f"{k}:下界{v['simultaneous_lower']:.3f}>{v['limit']}" for k,v in b['tests'].items() if v['simultaneous_lower']>v['limit']]
 lab={'supported_pass_under_t_assumptions':'在t假设下支持通过','supported_fail_under_t_assumptions':'在t假设下支持未达标','boundary_or_insufficient_evidence':'边界/证据不足'}[b['evidence']]
 lines.append(f"|{gl(g)}|{b['point_pass_seeds']}|{lab}|{'; '.join(bad) if bad else '无；不代表全部上界已通过'}|")
lines+=['','主要同时区间控制的是这100项种子平均指标；其他探索性分组的普通95%区间不作同等多重检验保证。支持通过还要求10个种子的原门槛点估计均通过；这也不保证每个未来轨迹或所有连续方向都通过。','',
 '## 四、瞬态峰值、持续时间与恢复','','事件基于已进入物理积分后的观测时刻：Release1601、Reverse3001，另加入Release/Reverse/Full3D加载启动1251/2451/4851。真值变化不做最佳滞后对齐。每个事件有效98个非零目标环境；释放按切换前力、启动按切换后力确定有效环境。','',
 '|事件|前100ms向量峰值的环境P95（N，跨种子均值[区间]）|前100ms大小峰值的环境P95 N|1s向量峰值环境P95 N|>1.5N总时长均值 s|>1.5N最长连续时长均值 s|','|---|---|---|---|---|---|']
for en,r in T.items():lines.append(f"|{en}|{st(r['peak_vector_100ms_N']['p95'])}|{st(r['peak_mag_100ms_N']['p95'])}|{st(r['peak_vector_1s_N']['p95'])}|{st(r['exceed_vector_1p5N_total_s']['mean'])}|{st(r['exceed_vector_1p5N_longest_s']['mean'])}|")
lines+=['','|事件|100ms最大向量误差 N|100ms最大大小误差 N|1s最大向量误差 N|100ms峰值角误差的环境P95均值 °|100ms最大角误差 °|','|---|---:|---:|---:|---:|---:|']
for en,r in T.items():
 angle_values=[v for seed in range(30101,30111) for v in runs[f'selected_{seed}']['transients'][en]['per_environment']['peak_angle_100ms_deg'] if v is not None]
 angle_p95=[runs[f'selected_{seed}']['transients'][en]['summary']['peak_angle_100ms_deg']['p95'] for seed in range(30101,30111)];angle_p95=[x for x in angle_p95 if x is not None]
 lines.append(f"|{en}|{r['peak_vector_100ms_N']['max']['max']:.3f}|{r['peak_mag_100ms_N']['max']['max']:.3f}|{r['peak_vector_1s_N']['max']['max']:.3f}|{fmt(sum(angle_p95)/len(angle_p95)) if angle_p95 else '—'}|{fmt(max(angle_values)) if angle_values else '—'}|")
lines+=['','|事件|>1.5N总时长的环境P95均值 s|>1.5N总时长最大值 s|>1.5N最长连续时长的环境P95均值 s|>1.5N最长连续时长最大值 s|','|---|---:|---:|---:|---:|']
for en,r in T.items():lines.append(f"|{en}|{r['exceed_vector_1p5N_total_s']['p95']['mean']:.3f}|{r['exceed_vector_1p5N_total_s']['max']['max']:.3f}|{r['exceed_vector_1p5N_longest_s']['p95']['mean']:.3f}|{r['exceed_vector_1p5N_longest_s']['max']['max']:.3f}|")
lines+=['','“峰值的环境P95”先在每环境时间窗内取峰值，再取该种子98环境的P95，最后对10种子汇总；不等同于把所有瞬态帧直接取P95。逐事件的真正最大值、参考力归一化峰值、误差积分保存在CSV/JSON，未隐藏极端值。','',
 '恢复条件沿用旧诊断：向量误差≤1.5N，真实力≥2N时另需方向≤10°，连续10个控制帧。以实际施力起点计时：首个受力观测在20ms后；恢复起点为(首个满足条件的观测序号+1)×20ms，确认时刻为(序号+10)×20ms。10帧覆盖200ms的积分区间，首末观测相隔180ms，不能把这两种时间混写。逐环境JSON保留旧“首个受力观测为0”的恢复起点供核对。此时间为仿真控制时间，不含实机传感器采样、通信和推理调度延迟。时间采样0.02s，恢复起点具有0–20ms离散分辨率限制；为在2s窗内完成10帧验证，最晚可确认的恢复起点为1.82s。下面的中位数/P95以全部98事件计算，未恢复以右删失保留，经验分位数为排序后第ceil(q×n)个、未恢复作无穷大；达到比例不足时记“未达到”，不使用只统计已恢复样本的乐观分位数。','',
 '|事件|两秒窗内确认恢复数，均值/98|未恢复数，均值/98|无重置且恢复数，均值/98|各种子恢复起点中位数 s|各种子恢复起点P95 s|','|---|---|---|---|---|---|']
for en,r in T.items():lines.append(f"|{en}|{r['recovered']['mean']:.2f}|{r['censored']['mean']:.2f}|{r['recovered_without_reset']['mean']:.2f}|"+', '.join(ftime(x) for x in r['settling_start_median_s_all_events'])+'|'+', '.join(ftime(x) for x in r['settling_start_p95_s_all_events'])+'|')
lines+=['','### 恢复预算网格（只作敏感性，不批准响应时间）','','每格为满足该向量误差、方向≤10°（适用时）、连续10帧确认的事件比例，先逐种子再等权平均。列为从实际施力起点计算的恢复**起点**期限，10帧的最后确认观测比恢复起点晚0.18s；2s列也要求在原2s窗口内完成确认。','']
for en,r in T.items():
 lines += [f'#### {en}','','|向量误差预算 N|0.1s|0.2s|0.3s|0.5s|1s|2s|','|---|---:|---:|---:|---:|---:|---:|']
 for b,cols in r['budget_grid'].items():lines.append('|'+b+'|'+'|'.join(f"{cols[t]['mean']:.1f}%" for t in ['0.1','0.2','0.3','0.5','1','2'])+'|')
 lines.append('')
lines+=['## 五、成对改进与闭环终止的不确定性','','以下为候选减去240轮对照，负值表示误差或终止减少；每对同种子同GPU，模型顺序预先平衡随机。','','|范围|大小RMSE差 N|方向P95差 °|Fz RMSE差 N|','|---|---|---|---|']
for g in ['overall','force/low','force/high','ramp','release','reverse','sweep','full3d']:
 r=a['paired_differences'][g];lines.append(f"|{gl(g)}|{st(r['mag_rmse'])}|{st(r['angle_p95'])}|{st(r['fz_rmse'])}|")
lines+=['','|种子|240轮终止|6320轮终止|差值|240轮受影响环境|6320轮受影响环境|','|---|---:|---:|---:|---:|---:|']
for s in range(30101,30111):
 b=runs[f'start_{s}']['terminations'];v=runs[f'selected_{s}']['terminations'];lines.append(f"|{s}|{b['count']}|{v['count']}|{v['count']-b['count']}|{b['unique_envs']}|{v['unique_envs']}|")
lines += ['',f"终止次数成对差：{st(terms['paired_difference'])}；判读：`{terms['interpretation']}`。区间跨0表示差异证据不足，不能解读为等效或非劣。允许的非劣界限尚未确定，也不能由本次结果反推。",'',
 '每条轨迹暴露量为112×120s=13440环境秒=3.7333环境小时，两模型各10条=37.3333环境小时。原始结果同时提供每环境小时终止率、唯一受影响环境、原因和同一环境反复重置次数。反复终止不是独立事故样本，不能对每次终止当独立样本计算置信区间。','',
 '## 六、数值、时间与测量不确定性边界','',
 f"20条轨迹逐个审计480000个物理子步。最大实际施加外力差为{max(r['audits']['runtime']['physics_audit']['max_wrench_error_N'] for r in runs.values()):.9g}N。历史缓存重建均严格一致，在线重叠缓存误差为0。CPU Torch与在线GPU最大重放差为{max(r['audits']['torch_cpu_online_replay_max_N'] for r in runs.values()):.9g}N；Torch/ONNX最大差为{a['onnx_parity']['max_N']:.9g}N。",'',
 '观测随机扰动配置记录为' + str(sorted({str(r['audits']['runtime']['runtime_contract']['observation_noise']) for r in runs.values()})) + '，具体来源保存在各轨迹元数据。控制周期来源于环境配置0.005s物理步长×4次积分=0.020s；配置与采集器哈希已保存。统计区间描述的是该仿真分布下跨种子变动，数值审计描述实现一致性，二者均不等于真实力传感器标定不确定度、参数辨识误差或仿真到实机差异。标签为刚完成积分的实际世界力旋转到观测时刻机身坐标，不是下一步力命令。仿真施加力是审计过的数字参考，不是经过实体计量溯源的传感器测量。未构造缺少来源的“合成测量不确定度”。','',
 '## 七、各新鲜种子的完整指标记录','']
for name in ['selected','start']:
 lines += [f'### {name}','','|种子/分组|大小MAE/RMSE N|分量MAE [Fx,Fy,Fz] N|分量RMSE [Fx,Fy,Fz] N|方向平均/P95 °|大小相对RMS %|向量相对RMS %|原绝对门槛点判定|','|---|---|---|---|---|---|---|---|']
 for seed in range(30101,30111):
  for g,r in runs[f'{name}_{seed}']['groups'].items():
   if not r['n']:continue
   lines.append(f"|{seed}/{gl(g)}|{fmt(r['mag_mae'])}/{fmt(r['mag_rmse'])}|{vec(r['component_mae'])}|{vec(r['component_rmse'])}|{fmt(r['angle_mean'])}/{fmt(r['angle_p95'])}|{fmt(r['relative_mag_rms_pct'])}|{fmt(r['relative_vector_rms_pct'])}|{'通过' if r['absolute_pass'] else '未通过'}|")
 lines.append('')
lines+=['## 八、范围、依据和留存','',
 '本次是原diagnostic控制契约、speed_scale=1.0和已声明phase-conditioned residual gate下的MuJoCo复验。每对模型施力计划逐元素哈希一致，但模型会导致各自闭环状态演化。10个种子仍沿用相同曲线族，不是所有方向/姿态/工况的穷举；本轮新数据现已被分析，后续调优需另留独立盲测。','',
 '正式合格限值仍需从应用允许力误差、控制器可容忍响应时间与错误接受/拒绝风险形成书面预算。当前用户明确未确定，因此没有把5/10/20%或0.1–2s任一档当作正式新标准。原绝对门槛失败如实保留，未用区间或新分组把已有失败改判为通过。','',
 '参考：[ISO/IEC Guide 98-4:2012公开摘要](https://www.iso.org/standard/50465.html)，用于区分规定限值、接受判定和风险；不是本项目1N/1.5N/5°/10°阈值来源，也不是本轮统计方法已获得ISO认证的证明。','',
 '方法工具：K-Dense experimental-design / scientific-critical-thinking。Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)，核对2026-09-02修订记录。','',
 '桌面配套数据：[逐种子指标]('+str(D/'力估计器新版重复验收数据/per_seed_metrics.csv')+')；[逐环境瞬态事件]('+str(D/'力估计器新版重复验收数据/transient_events.csv')+')；[完整汇总JSON]('+str(D/'力估计器新版重复验收数据/summary.json')+')。结果目录：`/home/zy/old_data/lead_clean_20260916/source/logs/rsl_rl/go2_velocity/acceptance_v2/`；原始轨迹：`logs/acceptance_v2/`。固定方案、分配、哈希、所有逐种子JSON和配套CSV一并保留。','',
 'TensorBoard：[验收记录](http://111.0.18.168:6013/#timeseries)，筛选`acceptance_v2_physical_`。普通95%区间与主要100项同时界限在JSON中分别保存，不相互替代。时间原点澄清记录保存在TIME_REFERENCE_NOTE.md：旧观测起点结果与源文件保留在observation_origin_archive目录，正式输出为timing_contract_version=2；原始轨迹和误差成绩未变。','']
bundle=D/'力估计器新版重复验收数据';bundle.mkdir(exist_ok=True)
for f in ['per_seed_metrics.csv','transient_events.csv','summary.json']:
 shutil.copy2(P/f,bundle/f)
text='\n'.join(lines)+'\n';out=D/'力估计器新版重复验收报告.md';out.write_text(text);(P/out.name).write_text(text)
main=D/'力估计器MuJoCo全范围验收报告.md';backup=D/'力估计器MuJoCo全范围验收报告_新版重复验收前备份.md'
if not backup.exists():shutil.copy2(main,backup)
main.write_text(text+'\n---\n\n# 历史记录（以下不代表本次新方案的正式判定）\n\n'+backup.read_text())
print('REPORT_WRITTEN',out,'characters',len(text))
