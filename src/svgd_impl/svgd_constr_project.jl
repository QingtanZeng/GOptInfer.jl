println(Base.active_project())

using LinearAlgebra
using Statistics
using Random
using Printf

ENV["GKSwstype"] = "100"
using Plots
gr() # 显式确认使用 GR 后端

"""
SVGD Kernel Function using RBF
计算核矩阵 K 以及核函数关于第一个变量的梯度 sum_j ∇_{x_j} k(x_j, x_i)
"""
function proj_svgd_kernel(X, h=-1.0)
    # X: n_particles x dim
    n = size(X, 1)
    
    # 计算成对欧氏距离的平方 (Pairwise squared Euclidean distances)
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    sum_sq = sum(X.^2, dims=2)
    # pairwise_dists_sq[i, j] = ||x_i - x_j||^2
    pairwise_dists_sq = sum_sq .+ sum_sq' .- 2 .* (X * X')
    
    # 启发式带宽选择: h = median(||x_i - x_j||^2) / log(n) 
    if h < 0
        median_dist_sq = median(pairwise_dists_sq)
        # 加上一个小量避免 log(1)=0 的情况
        h = median_dist_sq / log(n + 1e-6) 
    end
    
    # RBF Kernel: k(x, x') = exp(-||x - x'||^2 / h) 
    Kxy = exp.(-pairwise_dists_sq ./ h)

    # 计算梯度项 ∇_{x_j} k(x_j, x_i)
    # 基础 RBF 梯度: -2/h * (x_j - x_i) * k
    # 但我们需要将其投影。因为 P 是常数， ∇(P k P') = P * ∇k * P'
    # 简化: 我们在计算 update 时统一左乘 P 即可，
    # 这里只返回标量核矩阵 K_base 和 基础梯度 dx_k
    
    # 计算排斥力梯度的向量化实现
    # 我们需要计算: sum_j ∇_{x_j} k(x_j, x_i)
    # RBF导数: ∇_{x_j} k(x_j, x_i) = -2/h * (x_j - x_i) * k(x_j, x_i)
    
    # sum_k[i] = sum_j k(x_i, x_j)
    sum_k = sum(Kxy, dims=2)
    
    # 这一步计算 sum_j k(x_j, x_i) * (x_j - x_i)
    # = (sum_j k_{ji} x_j) - (sum_j k_{ji}) * x_i
    grad_k_part = (Kxy * X) .- (sum_k .* X)
    
    # 乘以系数 -2/h
    dx_kxy = -(2 / h) .* grad_k_part
    
    return Kxy, dx_kxy
end

"""
约束修正步 (Newton-like step)
将粒子拉回 Ax = b 平面 [cite: 625]
"""
function constraint_correction(X, A, b)
    # 对于每个粒子 x: x_new = x - A_pinv * (Ax - b)
    # 计算残差: (Ax - b)
    residuals = (A * X' .- b)' # n x 2
    
    # 计算修正量: - A_pinv * residual
    # distinct per particle
    corrections = -(pinv(A) * residuals')'
    return corrections
end

"""
SVGD Update Step
对应算法 1 中的核心更新公式 [cite: 113]
"""
function constrained_svgd_step(X, P, A, b, T, step_size, correction_strength=1.0)
    n = size(X, 1)
    
    # 1. 计算 Score Function: ∇ log p(x) = - ∇ f(x)
    # 对于 f(x) = x^2, ∇ f(x) = 2x, 所以 score = -2x
    score_val = -2/T .* X 
    
    # 2. 计算核矩阵和排斥力梯度
    Kxy, dx_kxy = proj_svgd_kernel(X)
    
    # 3. 组合最终梯度 phi
    # phi(x) = 1/n sum_j [ k(x_j, x) * score(x_j) + ∇_{x_j} k(x_j, x) ]
    # 矩阵乘法 Kxy * score_val 完成了对 score 的加权求和
    phi = (Kxy * score_val .+ dx_kxy) ./ n

    # 4. 投影梯度 (O-SVGD 核心步骤 [cite: 591])
    # 将更新方向投影到约束的切空间: P * phi
    phi_projected = phi * P'

    # 5. 计算约束修正力 (Constraint Correction)
    # 只有当粒子偏离平面时才生效
    phi_correction = constraint_correction(X, A, b)

    # 6. 更新位置
    # 组合: 切向移动(优化目标) + 法向移动(满足约束)
    X_new = X .+ (step_size .* phi_projected) .+ (correction_strength .* phi_correction)
    
    # 7. 应用不等式约束 (Bounds Constraint)
    # 直接投影 (Clipping) [cite: 757]
    X_new = max.(X_new, 0.0)
    
    return X_new
end

"""
在当前图上绘制线性约束 A*x = b
支持 1x2 的 A (直线) 或 2x2 的 A (点)
"""
function plot_constraints!(A, b, x_range=(-20, 20))
    # 1. 遍历每一行，画出对应的直线
    colors = [:cyan, :orange] # 为不同的约束线指定颜色
    for i in 1:size(A, 1)
        a1, a2 = A[i, 1], A[i, 2]
        val = b[i]
        
        # 定义画线的函数 y = f(x)
        # 避免除以零 (处理垂直线情况)
        if abs(a2) > 1e-6 
            # y = (b - a1*x) / a2
            plot!(x -> (val - a1 * x) / a2, x_range[1], x_range[2], 
                color=colors[mod1(i, end)], 
                linestyle=:dash, 
                linewidth=1.5,
                label="Constraint $i: $(round(a1,digits=1))x + $(round(a2,digits=1))y = $val"
            )
        else
            # x = b / a1 (垂直线)
            vline!([val / a1], color=colors[mod1(i, end)], linestyle=:dash, label="Constraint $i")
        end
    end

    # 2. 如果是满秩矩阵，画出唯一交点 (可行解)
    if size(A, 1) == size(A, 2) && det(A) != 0
        sol = A \ b
        scatter!([sol[1]], [sol[2]], 
            color=:green, 
            markershape=:star5, 
            markersize=12, 
            label="Feasible Solution $(round.(sol', digits=2))",
            markerstrokewidth=1, markerstrokecolor=:black
        )
    end
end


function main()
    # === 1. 问题定义 ===
    # 目标: min ||x||^2 -> log p(x) = -x'x
    # Score function: ∇ log p(x) = -2x

    # 等式约束: Ax = b
    A = [1.0 1.0; 1.0 1.0]
    b = [2.0, 2.0]
    # 计算投影矩阵 P (对于线性约束，P是常数)
    # P = I - A' * inv(A * A') * A
    # 作用: 将向量投影到 Ax=0 的零空间中
    A_pinv = pinv(A)
    P = I - A_pinv * A


    # 1. 参数设置
    Random.seed!(42)
    n_particles = 50
    dim = 2
    n_iter = 50       # 可以适当增加迭代次数
    step_size = 0.8   # 增加步长以便更快看到移动
    Eqstep_size = 0.9
    T=1

    # 2. 初始化 (局部变量，类型稳定)
    X = randn(n_particles, dim) .* 40 .+ kron(ones(n_particles, 1), [-15 -15]) 
    history = []
    push!(history, copy(X))
    
    println("开始计算...")
    
    # 3. 使用 @animate 宏录制动画
    # 这样不会在每一步都去调用 display 卡住主线程
    mean_i_lst = mean(X, dims=1)
    std_i_lst = std(X, dims=1)
    println("均值 (Empirical Mean): $mean_i_lst")
    println("标准差 (Empirical Std): $std_i_lst")
    for i in 1:n_iter
        # 计算步骤 (耗时极短)
        push!(history, copy(X))
        println("-"^50)
        println("Iter $i starts.")
        println("step_size: $step_size;  Eqstep_size: $Eqstep_size")
        
        T=max(0.95^(i-1), 0.5) 

        X = constrained_svgd_step(X, P, A, b, T, step_size, Eqstep_size)

        mean_i = mean(X, dims=1)
        std_i = std(X, dims=1)
        println("均值 (Empirical Mean): $mean_i")
        println("标准差 (Empirical Std): $std_i")

        S=A*X'.-b
        pres_eq = norm(S)/(norm(b[1:2])*n_particles)
        pres_eq_L = [norm(col) for col in eachcol(S)]/norm(b[1:2])
        pres_ineq = [norm(row) for row in eachrow(max.(0.0, 0 .- X')) ]./n_particles
        println("Mean Equality Error: $pres_eq;  Mean Inequality Error: $pres_ineq")
        Eqstep_size = min(max(0.5,pres_eq) ,0.9)


        meanchangeper = norm(mean_i - mean_i_lst) / norm(mean_i_lst)
        stdchangeper = norm(std_i - std_i_lst) / norm(std_i_lst)
        println("Mean Change Percent: $meanchangeper;  Std Change Percent: $stdchangeper")

        stdchangeper = norm(std_i - std_i_lst) / max(norm(std_i_lst),1)
        if(max(meanchangeper, stdchangeper) < 0.03 &&  maximum(pres_eq_L) < 3e-2 )
            println("收敛！")
            break;
        end

        meanchangeper = norm(mean_i - mean_i_lst) / max(norm(mean_i_lst),1)
        stdchangeper = norm(std_i - std_i_lst) / max(norm(std_i_lst),1)
        step_size = min(max(0.2, max(meanchangeper, stdchangeper)*10),0.9)

        mean_i_lst = mean_i;
        std_i_lst = std_i;
    end

    # 结果分析
    println("-"^30)
    println("优化后结果 (Target: Mean≈0, Std≈0.707):")
    final_mean = mean(X, dims=1)
    final_std = std(X, dims=1)
    println("最终均值 (Empirical Mean): ", final_mean)
    println("最终标准差 (Empirical Std): ", final_std)
    println("Results: $X")
    S=A*X'.-b
    eqerr = [norm(col) for col in eachcol(S)]
    println("Mean Equality absolute Error: $eqerr; ")
    println("-"^30)

    println("计算完成，正在生成 GIF...")
    # 4. 保存为 GIF 文件 (首次运行这一步可能需要几十秒编译 ffmpeg 接口)
    # 绘图步骤 (在内存中构建帧)
    output_dir = "test/svgd_frames"
    mkpath(output_dir)  # 如果文件夹不存在会自动创建
    for i in 1:2:length(history)
        X = history[i]
        # meanval=mean(X, dims=1)
        # stdval=std(X, dims=1)
        # xleft=meanval[1]-2*stdval[1]
        # xright=meanval[1]+3*stdval[1]
        # yleft=meanval[2]-2*stdval[2]
        # yright=meanval[2]+3*stdval[2]

        # # 3. 创建绘图对象
        # p = scatter(
        #     X[:, 1], X[:, 2], 
        #     color = :purple, 
        #     label = "Particles",
        #     # 视窗跟随粒子群移动
        #     xlims = (xleft, xright), 
        #     ylims = (yleft, yright),
        #     aspect_ratio = :equal,
        #     title = "SVGD Optimization (Iter $i)",
        #     markerstrokewidth = 0, markersize = 3, alpha = 0.7
        # )

        # --- 修改方案 B：动态视窗，但强制以粒子均值为中心 (推荐用于观察局部细节) ---
        mean_x, mean_y = mean(X[:,1]), mean(X[:,2])
        # 找出最大的扩散范围，保证视窗是正方形的，且留有足够余地
        # 设定最小视窗半径为 1.0，防止收敛后视窗缩得太小看不清直线
        radius = max(3 * maximum(std(X, dims=1)), 1.0) 
        
        xleft, xright = mean_x - radius, mean_x + radius
        yleft, yright = mean_y - radius, mean_y + radius
    
        p = scatter(
            X[:, 1], X[:, 2], 
            color = :purple, 
            label = "Particles",
            xlims = (xleft, xright), 
            ylims = (yleft, yright),
            aspect_ratio = :equal, # 保持 x 和 y 轴比例 1:1
            title = "SVGD Iter $i",
            markerstrokewidth = 0, markersize = 4, alpha = 0.8
        )

        # 叠加目标中心 (1,1)
        scatter!([1], [1], color=:red, shape=:star5, ms=10, label="True Solution (1,1)")

        # 4. [关键步骤] 叠加约束线和交点
        # 注意：传入的 range 只需要覆盖当前的 xleft/xright 即可
        # 但为了简单，传一个足够大的固定范围覆盖全局即可，Plots 会自动裁剪
        # plot_constraints!(A, b, (xleft-10, xright+10))

        # 5. 保存
        filename = joinpath(output_dir, @sprintf("iter_%04d.png", i))
        savefig(p, filename)
        
        print("\r正在保存: $filename")
    
    end
end

# 执行主函数
main()
