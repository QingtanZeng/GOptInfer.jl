import re
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 原始数据 (Raw Data)
# ==========================================
log_data = """
开始计算...
均值 (Empirical Mean): [-14.748159382729051 -24.05168474833777]
标准差 (Empirical Std): [38.71787047546534 38.02380978701859]
--------------------------------------------------
Iter 1 starts.
step_size: 0.8;  Eqstep_size: 0.9
均值 (Empirical Mean): [10.185957276468065 6.659926290221292]
标准差 (Empirical Std): [15.802785237662665 14.373207471785975]
Mean Equality Error: 1.6308212372471351;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 1.402138288336426;  Std Change Percent: 0.6068357026786722
--------------------------------------------------
Iter 2 starts.
step_size: 0.9;  Eqstep_size: 0.9
均值 (Empirical Mean): [5.159005108779615 3.463465276526159]
标准差 (Empirical Std): [7.723366054616867 7.688864899919046]
Mean Equality Error: 0.7898060535881184;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.48949540554687315;  Std Change Percent: 0.49088408387786964
--------------------------------------------------
Iter 3 starts.
step_size: 0.9;  Eqstep_size: 0.7898060535881184
均值 (Empirical Mean): [3.113251403152879 2.1975115573668322]
标准差 (Empirical Std): [4.3671217338402455 4.621184387171354]
Mean Equality Error: 0.43271799606810535;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.3871681875114202;  Std Change Percent: 0.4172262044677279
--------------------------------------------------
Iter 4 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [2.390905843961667 1.7976248217517286]
标准差 (Empirical Std): [3.1136990159161892 3.4211712410869026]
Mean Equality Error: 0.30157705785788064;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.21666569075671108;  Std Change Percent: 0.272914313413107
--------------------------------------------------
Iter 5 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.9194838808607861 1.5254024730232068]
标准差 (Empirical Std): [2.27097434048621 2.567308390779563]
Mean Equality Error: 0.21060123739894848;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.18198584751256008;  Std Change Percent: 0.25933938112302074
--------------------------------------------------
Iter 6 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.6031712765020945 1.3376767923601705]
标准差 (Empirical Std): [1.6950105253835723 1.957345578813949]
Mean Equality Error: 0.14686516787148227;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.15002273096338778;  Std Change Percent: 0.2447551944890551
--------------------------------------------------
Iter 7 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.3828152218145027 1.2252171942645644]
标准差 (Empirical Std): [1.298853807477906 1.5090003895989645]
Mean Equality Error: 0.10250701178586623;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.11848668036809873;  Std Change Percent: 0.23106695296855678
--------------------------------------------------
Iter 8 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.2249350937733297 1.1710852748872425]
标准差 (Empirical Std): [1.0290352910116327 1.1745904371360085]
Mean Equality Error: 0.07174797314536246;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.09033853163935134;  Std Change Percent: 0.21581471706320063
--------------------------------------------------
Iter 9 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.1147866389436643 1.1427391254309232]
标准差 (Empirical Std): [0.8475288367782998 0.9359665046613032]
Mean Equality Error: 0.050001337807982604;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.06711474844897665;  Std Change Percent: 0.19198963723135698
--------------------------------------------------
Iter 10 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.0599756529688695 1.1066287402585704]
标准差 (Empirical Std): [0.7201052151674516 0.7743556210567223]
Mean Equality Error: 0.03448843498364048;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.04111470027187527;  Std Change Percent: 0.16299014769464684
--------------------------------------------------
Iter 11 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.0326352545802442 1.072887340038107]
标准差 (Empirical Std): [0.6191574256349724 0.6554190590374235]
Mean Equality Error: 0.02332405972198986;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.02834021884491432;  Std Change Percent: 0.14752727746788077
--------------------------------------------------
Iter 12 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.021214716720826 1.0436094906314468]
标准差 (Empirical Std): [0.5551552164360056 0.5819460624364229]
Mean Equality Error: 0.01534324493885494;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.021104299204787565;  Std Change Percent: 0.10807143589679252
--------------------------------------------------
Iter 13 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.016879691986694 1.0213430811035502]
标准差 (Empirical Std): [0.48920829339203903 0.5098313135321532]
Mean Equality Error: 0.009728462851264172;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.015535853473052426;  Std Change Percent: 0.12150283911497921
--------------------------------------------------
Iter 14 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.0014962982237747 1.0203618429418424]
标准差 (Empirical Std): [0.500960606944016 0.5133266562028582]
Mean Equality Error: 0.005903131437331462;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.010695378425791009;  Std Change Percent: 0.017352770001338624
--------------------------------------------------
Iter 15 starts.
step_size: 0.2;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.0010353932409908 1.0130083748813674]
标准差 (Empirical Std): [0.4533362573822917 0.46298636283770284]
Mean Equality Error: 0.004104131868178095;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.005153342848691778;  Std Change Percent: 0.09661464211525149
--------------------------------------------------
Iter 16 starts.
step_size: 0.692980793908981;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.0017234660096372 1.0058360380188882]
标准差 (Empirical Std): [0.40480339971240165 0.4113820361883354]
Mean Equality Error: 0.002292671687703308;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.005059276870534526;  Std Change Percent: 0.10932691260581973
--------------------------------------------------
Iter 17 starts.
step_size: 0.7084098250688242;  Eqstep_size: 0.5
均值 (Empirical Mean): [0.9925362671823291 1.011243484831933]
标准差 (Empirical Std): [0.46261475154850434 0.46526589228716475]
Mean Equality Error: 0.001146335843851655;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.007509672953371397;  Std Change Percent: 0.1369304105095288
--------------------------------------------------
Iter 18 starts.
step_size: 0.7902924995972695;  Eqstep_size: 0.5
均值 (Empirical Mean): [0.9945104811446719 1.007379394862459]
标准差 (Empirical Std): [0.33739926662587655 0.3391448962952154]
Mean Equality Error: 0.0005731679219258274;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.003062361070504592;  Std Change Percent: 0.2708722160061554
--------------------------------------------------
Iter 19 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [0.9534791960080813 1.0594361594108412]
标准差 (Empirical Std): [0.6752150485470805 0.6548619466755543]
Mean Equality Error: 0.0029807740127120214;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.04682414186987919;  Std Change Percent: 0.966535831485402
--------------------------------------------------
Iter 20 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.1391420591017742 0.8673156186076864]
标准差 (Empirical Std): [0.27637412971480946 0.2664364479751888]
Mean Equality Error: 0.001490387006356003;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.18744763194646352;  Std Change Percent: 0.5918778767193625
--------------------------------------------------
Iter 21 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [0.994409710948466 1.1266567546098674]
标准差 (Empirical Std): [0.8215032202264567 0.8278286935901268]
Mean Equality Error: 0.015537672796304275;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.2074354408736048;  Std Change Percent: 2.03838222031739
--------------------------------------------------
Iter 22 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.0716114086818649 0.9889218240973012]
标准差 (Empirical Std): [0.3300939434917917 0.35848009759915]
Mean Equality Error: 0.00776883639815212;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.10507228449111602;  Std Change Percent: 0.5826626161410937
--------------------------------------------------
Iter 23 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [0.9169469686932044 1.1133196476963783]
标准差 (Empirical Std): [0.5280994286946451 0.5241276373215763]
Mean Equality Error: 0.0038844181990760505;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.13611673071220978;  Std Change Percent: 0.5297614210719569
--------------------------------------------------
Iter 24 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.1250830101801446 0.8900502980146462]
标准差 (Empirical Std): [0.3463720373103784 0.3455109681756719]
Mean Equality Error: 0.0019422090995380083;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.21163013709085202;  Std Change Percent: 0.3424687803981955
--------------------------------------------------
Iter 25 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [0.9930916472257443 1.0144750068716506]
标准差 (Empirical Std): [0.42361956481248747 0.42522897659859765]
Mean Equality Error: 0.0009711045497689857;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.1264434771178921;  Std Change Percent: 0.22689508682172302
--------------------------------------------------
Iter 26 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.0609934165693227 0.9427899104793742]
标准差 (Empirical Std): [0.3128773959372552 0.3139478760937606]
Mean Equality Error: 0.0004855522748844662;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.06955196569575814;  Std Change Percent: 0.2615584684142314
--------------------------------------------------
Iter 27 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [0.9595960399084004 1.0438728377640853]
标准差 (Empirical Std): [0.45901244775374084 0.4559230808711106]
Mean Equality Error: 0.0008251313676430511;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.10087376329240581;  Std Change Percent: 0.45968131541928775
--------------------------------------------------
Iter 28 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.1335174624980562 0.8682169763381862]
标准差 (Empirical Std): [0.4763211986881273 0.4749540590885142]
Mean Equality Error: 0.0004125656838215143;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.17433398041406883;  Std Change Percent: 0.03976270237705634
--------------------------------------------------
Iter 29 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [0.9458406804562128 1.0556140362521873]
标准差 (Empirical Std): [0.2869854633811352 0.28327858326871563]
Mean Equality Error: 0.0004946382968980369;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.18575017297264468;  Std Change Percent: 0.40053391327130905
--------------------------------------------------
Iter 30 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.0022490444344125 0.998478313919787]
标准差 (Empirical Std): [0.422956225322338 0.42188745583533555]
Mean Equality Error: 0.00024731914844901574;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.05664680592061197;  Std Change Percent: 0.48150765943943896
--------------------------------------------------
Iter 31 starts.
step_size: 0.9;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.0065933267983616 0.9937703523787377]
标准差 (Empirical Std): [0.37767050684458936 0.3770872867501378]
Mean Equality Error: 0.00012365957422450326;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.004528118408205374;  Std Change Percent: 0.10663170713855219
--------------------------------------------------
Iter 32 starts.
step_size: 0.6370126724099147;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.003998010240171 0.9961838293483782]
标准差 (Empirical Std): [0.3306864154095868 0.3303632179948372]
Mean Equality Error: 6.182978711225427e-5;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.0025055395055483855;  Std Change Percent: 0.12415703920551341
--------------------------------------------------
Iter 33 starts.
step_size: 0.6626193061647644;  Eqstep_size: 0.5
均值 (Empirical Mean): [1.0210247328538762 0.9790661869403979]
标准差 (Empirical Std): [0.3242566134040487 0.3241070466068606]
Mean Equality Error: 3.091489355612087e-5;  Mean Inequality Error: [0.0, 0.0]
Mean Change Percent: 0.017070560702545784;  Std Change Percent: 0.01919244699004523
"""

# ==========================================
# 2. 数据解析 (Parsing)
# ==========================================
iters = []
means = []   # List of [val1, val2]
stds = []    # List of [val1, val2]
eq_errors = []

# --- Step A: 提取初始状态 (Iter 0) ---
# 初始块在 "开始计算..." 和 "Iter 1" 之间
init_block_match = re.search(r"开始计算\.\.\.(.*?)Iter 1", log_data, re.DOTALL)
if init_block_match:
    init_text = init_block_match.group(1)
    
    # 提取初始均值
    mean_match = re.search(r"均值 \(Empirical Mean\): \[(.*?)\]", init_text)
    if mean_match:
        vals = [float(x) for x in mean_match.group(1).split()]
        means.append(vals)
        
    # 提取初始标准差
    std_match = re.search(r"标准差 \(Empirical Std\): \[(.*?)\]", init_text)
    if std_match:
        vals = [float(x) for x in std_match.group(1).split()]
        stds.append(vals)
    
    iters.append(0)

# --- Step B: 循环提取每次迭代的数据 ---
lines = log_data.split('\n')
current_iter = None

for line in lines:
    line = line.strip()
    
    # 提取迭代号
    if line.startswith("Iter") and "starts" in line:
        parts = line.split()
        if parts[1].isdigit():
            current_iter = int(parts[1])
            iters.append(current_iter)
            
    # 提取均值
    elif line.startswith("均值"):
        # 防止重复添加初始均值 (当 current_iter 为 None 时表示还在初始块，已处理过)
        if current_iter is None:
            continue
            
        match = re.search(r"\[(.*?)\]", line)
        if match:
            vals = [float(x) for x in match.group(1).split()]
            means.append(vals)
            
    # 提取标准差
    elif line.startswith("标准差"):
        if current_iter is None:
            continue
            
        match = re.search(r"\[(.*?)\]", line)
        if match:
            vals = [float(x) for x in match.group(1).split()]
            stds.append(vals)
            
    # 提取 Mean Equality Error
    elif line.startswith("Mean Equality Error"):
        parts = line.split(';')
        for part in parts:
            if "Mean Equality Error" in part:
                val_str = part.split(':')[1].strip()
                eq_errors.append(float(val_str))

# --- Step C: 数据对齐检查与处理 ---
# 确保数据长度一致 (以 iters 长度为准)
min_len = min(len(iters), len(means), len(stds))
iters = iters[:min_len]
means = means[:min_len]
stds = stds[:min_len]

iters_arr = np.array(iters)
means_arr = np.array(means)  # shape (N, 2)
stds_arr = np.array(stds)    # shape (N, 2)
eq_errors_arr = np.array(eq_errors) 

print(f"Parsed {len(iters)} iterations.")
print(f"Means shape: {means_arr.shape}")
print(f"Eq Errors count: {len(eq_errors_arr)}")

# ==========================================
# 3. 绘图 (Plotting)
# ==========================================
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig.suptitle('Convergence Analysis: Mean, Std, and Equality Error', fontsize=16)

# --- Subplot 1: Empirical Mean ---
axs[0].plot(iters_arr, means_arr[:, 0], 'b-o', label='Mean Dim 1', markersize=4)
axs[0].plot(iters_arr, means_arr[:, 1], 'c-s', label='Mean Dim 2', markersize=4)
axs[0].set_ylabel('Value')
axs[0].set_title('Empirical Mean vs. Iteration')
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].legend()

# --- Subplot 2: Empirical Std (Modified to Log Scale) ---
# 使用 semilogy 来实现 Y 轴对数坐标
axs[1].semilogy(iters_arr, stds_arr[:, 0], 'm-o', label='Std Dim 1', markersize=4)
axs[1].semilogy(iters_arr, stds_arr[:, 1], 'r-s', label='Std Dim 2', markersize=4)
axs[1].set_ylabel('Value (Log Scale)')
axs[1].set_title('Empirical Std vs. Iteration (Log Scale)')
axs[1].grid(True, which="both", linestyle='--', alpha=0.7)
axs[1].legend()

# --- Subplot 3: Mean Equality Error ---
# Error 通常从 Iter 1 开始，所以取 iters[1:] 和对应的 error 数据
valid_err_len = min(len(iters_arr)-1, len(eq_errors_arr))
if valid_err_len > 0:
    axs[2].semilogy(iters_arr[1:1+valid_err_len], eq_errors_arr[:valid_err_len], 'g-^', label='Mean Equality Error', linewidth=2)

axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Error (Log Scale)')
axs[2].set_title('Mean Equality Error vs. Iteration')
axs[2].grid(True, which="both", linestyle='--', alpha=0.7)
axs[2].legend()


# 调整整体布局防止标签重叠
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('convergence_plot.png')