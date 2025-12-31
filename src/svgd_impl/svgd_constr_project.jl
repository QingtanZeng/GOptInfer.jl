println(Base.active_project())

using LinearAlgebra
using Statistics
using Random
using GLMakie

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
function constrained_svgd_step(X, P, A, b, step_size, correction_strength=1.0)
    n = size(X, 1)
    
    # 1. 计算 Score Function: ∇ log p(x) = - ∇ f(x)
    # 对于 f(x) = x^2, ∇ f(x) = 2x, 所以 score = -2x
    score_val = -2 .* X 
    
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

function run_makie_animation(history)
    # 1. 初始化数据为 Observable
    # 取第一帧数据作为初始状态
    points_node = Observable(Point2f.(history[1][:,1], history[1][:,2]))
    
    # 2. 创建画布和静态元素 (只创建一次！)
    fig = Figure(size = (600, 600))
    ax = Axis(fig[1, 1], 
        title = "SVGD Optimization", 
        limits = (-20, 20, -20, 20),
        aspect = DataAspect()
    )
    
    # 3. 绘制粒子 (绑定到 Observable)
    GLMakie.scatter!(ax, points_node, color = :purple, markersize = 10, label="Particles")
    # 绘制目标 (静态)
    GLMakie.scatter!(ax, [0], [0], color = :red, marker = :xcross, markersize = 20, label="Target")
    
    # 4. 录制动画
    # record 函数会自动高效地更新 Observable
    record(fig, "svgd_fast.gif", 1:length(history); framerate = 10) do i
        # 核心：只更新数据坐标，不重绘坐标轴
        X_current = history[i]
        points_node[] = Point2f.(X_current[:,1], X_current[:,2])
    end
    println("GLMakie 动画已保存。")
end


function main()
    # === 1. 问题定义 ===
    # 目标: min ||x||^2 -> log p(x) = -x'x
    # Score function: ∇ log p(x) = -2x

    # 等式约束: Ax = b
    A = [1.0 1.0; -1.0 1.0]
    b = [2.0, 0.0]
    # 计算投影矩阵 P (对于线性约束，P是常数)
    # P = I - A' * inv(A * A') * A
    # 作用: 将向量投影到 Ax=0 的零空间中
    A_pinv = pinv(A)
    P = I - A_pinv * A


    # 1. 参数设置
    Random.seed!(42)
    n_particles = 10
    dim = 2
    n_iter = 15       # 可以适当增加迭代次数
    step_size = 0.8   # 增加步长以便更快看到移动
    Eqstep_size = 0.9

    # 2. 初始化 (局部变量，类型稳定)
    X = randn(n_particles, dim) .* 20 .+ kron(ones(n_particles, 1), [10 -15]) 
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
        println("Iter $i starts.")
        println("step_size: $step_size;  Eqstep_size: $Eqstep_size")

        X = constrained_svgd_step(X, P, A, b, step_size, Eqstep_size)

        mean_i = mean(X, dims=1)
        std_i = std(X, dims=1)
        println("均值 (Empirical Mean): $mean_i")
        println("标准差 (Empirical Std): $std_i")

        S=A*X'.-b
        pres_eq = norm(S)/(norm(b[1:2])*n_particles)
        pres_ineq = [norm(row) for row in eachrow(max.(0.0, 0 .- X')) ]./n_particles
        println("Mean Equality Error: $pres_eq;  Mean Inequality Error: $pres_ineq")
        Eqstep_size = min(max(0.5,pres_eq) ,0.9)


        meanchangeper = norm(mean_i - mean_i_lst) / norm(mean_i_lst)
        stdchangeper = norm(std_i - std_i_lst) / norm(std_i_lst)
        println("Mean Change Percent: $meanchangeper;  Std Change Percent: $stdchangeper")

        stdchangeper = norm(std_i - std_i_lst) / max(norm(std_i_lst),1)
        if(max(meanchangeper, stdchangeper) < 0.05 && pres_ineq < 1e-2 &&  pres_eq < 3e-2 )
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

    println("计算完成，正在生成 GIF...")
    # 4. 保存为 GIF 文件 (首次运行这一步可能需要几十秒编译 ffmpeg 接口)
    # 绘图步骤 (在内存中构建帧)
    anim = @animate for i in 1:length(history)
        X = history[i]
        # 绘制粒子轨迹
        scatter(
            X[:, 1], X[:, 2], 
            color = :purple, 
            label = "Iter $i",
            xlims = (-20, 20), ylims = (-20, 20),
            aspect_ratio = :equal,
            title = "SVGD Optimization",
            markerstrokewidth = 0, markersize = 2, alpha = 0.7
        )
        scatter!([0], [0], color=:red, shape=:cross, ms=8, label="Target")
    end
    # 绘制目标中心
    mp4(anim, "src/svgd_impl/svgd_animation.mp4", fps = 10)
    println("动画已保存至当前目录: svgd_animation.gif")
end

# 执行主函数
main()



