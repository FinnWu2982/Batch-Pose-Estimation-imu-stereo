import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm, logm, inv
import matplotlib.pyplot as plt

# ==============================================================================
# SECTION 1: SE3 Math Library (工具库)
# ==============================================================================

# 请确保路径正确
MAT_PATH = r'D:\下载\dataset3.mat'  # 建议使用相对路径，或者改回你的绝对路径


def skew(v):
    """生成 3x3 反对称矩阵"""
    v = v.flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def hat(xi):
    """将 6维向量 映射到 se(3) 李代数"""
    xi = xi.flatten()
    rho, phi = xi[0:3], xi[3:6]
    Xi = np.zeros((4, 4))
    Xi[0:3, 0:3] = skew(phi)
    Xi[0:3, 3] = rho
    return Xi


def vee(Xi):
    """将 se(3) 映射回 6维向量"""
    phi = np.array([Xi[2, 1], Xi[0, 2], Xi[1, 0]])
    rho = Xi[0:3, 3]
    return np.hstack((rho, phi))


def compute_adjoint(T):
    """计算 Adjoint 矩阵 (6x6)"""
    C, r = T[0:3, 0:3], T[0:3, 3]
    Ad = np.zeros((6, 6))
    Ad[0:3, 0:3] = C
    Ad[0:3, 3:6] = skew(r) @ C
    Ad[3:6, 3:6] = C
    return Ad


def se3_inverse(T):
    """SE(3) 矩阵求逆"""
    C, r = T[0:3, 0:3], T[0:3, 3]
    T_inv = np.eye(4)
    T_inv[0:3, 0:3] = C.T
    T_inv[0:3, 3] = -C.T @ r
    return T_inv


def axis_angle_to_rot(psi):
    """轴角转旋转矩阵"""
    psi = np.array(psi).flatten()
    angle = np.linalg.norm(psi)
    if angle < 1e-9: return np.eye(3)
    axis = psi / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def compute_point_interaction_matrix(p_homo):
    """
    计算点对位姿的相互作用矩阵 (Point-to-Pose Jacobian helper)
    对应 p_dot 算子: p_dot * xi = xi_hat * p
    """
    p = p_homo.flatten()
    xyz = p[0:3]
    # homogeneous part is usually 1
    D = np.zeros((4, 6))
    D[0:3, 0:3] = np.eye(3) * p[3]
    D[0:3, 3:6] = -skew(xyz)
    return D


# ==============================================================================
# SECTION 2: 求解器核心类 (Levenberg-Marquardt Solver)
# ==============================================================================

class BatchSolver:
    def __init__(self, dataset_path):
        print(f"Initializing solver with data: {dataset_path}")
        try:
            data = sio.loadmat(dataset_path)
        except FileNotFoundError:
            print(f"Error: {dataset_path} not found.")
            return

        self.times = data['t'].flatten()
        self.vel_lin = data['v_vk_vk_i']
        self.vel_ang = data['w_vk_vk_i']
        self.gt_pos = data['r_i_vk_i']
        self.gt_rot = data['theta_vk_i']

        C_c_v = data['C_c_v']
        rho_v_c_v = data['rho_v_c_v'].flatten()

        # T_cv: Vehicle Frame 到 Camera Frame 的变换
        self.T_cv = np.eye(4)
        self.T_cv[0:3, 0:3] = C_c_v
        self.T_cv[0:3, 3] = -C_c_v @ rho_v_c_v

        fu, fv, cu, cv, b = [data[k].item() for k in ['fu', 'fv', 'cu', 'cv', 'b']]

        # 双目投影矩阵
        self.K_stereo = np.array([
            [fu, 0, cu, 0],
            [0, fv, cv, 0],
            [fu, 0, cu, -fu * b],
            [0, fv, cv, 0]
        ])

        self.landmarks = data['rho_i_pj_i']
        self.measurements = data['y_k_j']

        self.Q_diag_base = np.concatenate([data['v_var'].flatten(), data['w_var'].flatten()])
        self.R_cov_block = np.diag(data['y_var'].flatten())
        self.R_inv_block = np.linalg.inv(self.R_cov_block)

        self.last_full_cov = None

    def integrate_dead_reckoning(self, k_start, k_end, T_initial):
        """航迹推算 (右乘更新)"""
        num_steps = k_end - k_start + 1
        traj = [None] * num_steps
        traj[0] = T_initial.copy()

        for i in range(num_steps - 1):
            idx = k_start + i
            dt = self.times[idx + 1] - self.times[idx]
            xi = np.concatenate([self.vel_lin[:, idx], self.vel_ang[:, idx]]) * dt
            # Right Multiply for Body Frame
            traj[i + 1] = traj[i] @ expm(hat(xi))

        return traj

    def solve(self, k_start, k_end, initial_guess, prior_info, max_iter=15):
        """
        Levenberg-Marquardt 求解器
        返回: current_est, final_cov, full_cov
        """
        num_poses = k_end - k_start + 1
        num_motion = num_poses - 1
        dim = 6 * num_poses

        current_est = [T.copy() for T in initial_guess]
        final_cov = None

        # LM 参数
        lambda_val = 1e-2
        chi2_prev = None

        for iteration in range(max_iter):
            vals, rows, cols = [], [], []
            rhs_b = np.zeros(dim)
            total_chi2 = 0.0

            def push_block(mat, r_start, c_start):
                r_idx, c_idx = np.indices(mat.shape)
                vals.extend(mat.flatten())
                rows.extend((r_idx + r_start).flatten())
                cols.extend((c_idx + c_start).flatten())

            # 1. Prior
            push_block(prior_info, 0, 0)

            # 2. Motion
            for k in range(num_motion):
                global_idx = k_start + k
                dt = self.times[global_idx + 1] - self.times[global_idx]
                idx_curr, idx_next = 6 * k, 6 * (k + 1)

                T_curr, T_next = current_est[k], current_est[k + 1]
                xi_meas = np.concatenate([self.vel_lin[:, global_idx], self.vel_ang[:, global_idx]]) * dt
                T_meas = expm(hat(xi_meas))

                Q_k = np.diag(self.Q_diag_base * dt ** 2)
                W_mot = np.linalg.inv(Q_k)

                # e = log( T_curr^{-1} * T_next * T_meas^{-1} )
                T_rel_est = se3_inverse(T_curr) @ T_next
                err_se3 = logm(T_rel_est @ se3_inverse(T_meas))
                error = vee(err_se3).reshape(-1, 1)

                total_chi2 += 0.5 * (error.T @ W_mot @ error).item()

                J_curr = -compute_adjoint(se3_inverse(T_rel_est))
                J_next = np.eye(6)

                push_block(J_curr.T @ W_mot @ J_curr, idx_curr, idx_curr)
                push_block(J_curr.T @ W_mot @ J_next, idx_curr, idx_next)
                push_block(J_next.T @ W_mot @ J_curr, idx_next, idx_curr)
                push_block(J_next.T @ W_mot @ J_next, idx_next, idx_next)

                rhs_b[idx_curr:idx_curr + 6] -= (J_curr.T @ W_mot @ error).flatten()
                rhs_b[idx_next:idx_next + 6] -= (J_next.T @ W_mot @ error).flatten()

            # 3. Measurement
            fx = self.K_stereo[0, 0]
            fy = self.K_stereo[1, 1]
            cx = self.K_stereo[0, 2]
            cy = self.K_stereo[1, 2]
            b_val = -self.K_stereo[2, 3] / fx if fx != 0 else 0

            for k in range(num_poses):
                global_idx = k_start + k
                state_idx = 6 * k
                T_vi = current_est[k]
                T_iv = se3_inverse(T_vi)

                valid_mask = self.measurements[0, global_idx, :] != -1
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) == 0: continue

                for j in valid_indices:
                    y_meas = self.measurements[:, global_idx, j]
                    lm_w = np.append(self.landmarks[:, j], 1.0)

                    # World -> Vehicle -> Camera
                    pt_vehicle = T_iv @ lm_w
                    pt_cam = self.T_cv @ pt_vehicle
                    X, Y, Z = pt_cam[0], pt_cam[1], pt_cam[2]

                    if Z < 0.1: continue

                    inv_z = 1.0 / Z
                    inv_z2 = inv_z * inv_z

                    u_L = fx * X * inv_z + cx
                    v_L = fy * Y * inv_z + cy
                    u_R = fx * (X - b_val) * inv_z + cx
                    v_R = v_L

                    y_pred = np.array([u_L, v_L, u_R, v_R])
                    e_ij = y_meas.reshape(4, 1) - y_pred.reshape(4, 1)

                    # Huber Loss-like check
                    if np.linalg.norm(e_ij) > 200: continue

                    total_chi2 += 0.5 * (e_ij.T @ self.R_inv_block @ e_ij).item()

                    d_proj = np.zeros((4, 4))
                    d_proj[0, 0] = fx * inv_z;
                    d_proj[0, 2] = -fx * X * inv_z2
                    d_proj[1, 1] = fy * inv_z;
                    d_proj[1, 2] = -fy * Y * inv_z2
                    d_proj[2, 0] = fx * inv_z;
                    d_proj[2, 2] = -fx * (X - b_val) * inv_z2
                    d_proj[3, 1] = fy * inv_z;
                    d_proj[3, 2] = -fy * Y * inv_z2

                    J_geo = -compute_point_interaction_matrix(pt_vehicle)
                    G_ij = d_proj @ self.T_cv @ J_geo

                    w_G = self.R_inv_block @ G_ij
                    push_block(G_ij.T @ w_G, state_idx, state_idx)
                    rhs_b[state_idx: state_idx + 6] += (w_G.T @ e_ij).flatten()

            # 4. Solve (LM Damping)
            H_mat = sp.csc_matrix((vals, (rows, cols)), shape=(dim, dim))
            H_damp = H_mat + sp.diags([lambda_val] * dim)

            try:
                delta_x = spla.spsolve(H_damp, rhs_b)
            except:
                print("Linear solve failed, increasing lambda.")
                lambda_val *= 10
                continue

            norm_dx = np.linalg.norm(delta_x)
            print(f"Iter {iteration}: Chi2 = {total_chi2:.2e}, dx = {norm_dx:.4f}, lambda = {lambda_val:.1e}")

            if chi2_prev is not None and total_chi2 > chi2_prev:
                lambda_val *= 10.0  # Error increased, dampen more
            else:
                lambda_val = max(1e-9, lambda_val * 0.1)  # Error decreased, dampen less
                chi2_prev = total_chi2

            for k in range(num_poses):
                dx_k = delta_x[6 * k: 6 * k + 6]
                current_est[k] = current_est[k] @ expm(hat(dx_k))

            if norm_dx < 1e-3:
                print("Converged.")
                break

        # 5. Extract Covariance
        try:
            cov_matrix = np.linalg.inv(H_mat.toarray())
            self.last_full_cov = cov_matrix
            variances = np.diag(cov_matrix)
            final_cov = variances.reshape(num_poses, 6).T
        except:
            final_cov = np.zeros((6, num_poses))
            self.last_full_cov = np.eye(dim)

        return current_est, final_cov, self.last_full_cov


# ==============================================================================
# SECTION 3: 可视化与主流程
# ==============================================================================

def visualize_results(times, gt_pos, gt_rot, est_traj, est_cov, title_suffix=""):
    N = len(est_traj)
    err_pos = np.zeros((3, N))
    err_rot = np.zeros((3, N))
    sigma = np.sqrt(est_cov)

    for i in range(N):
        r_est = est_traj[i][0:3, 3]
        err_pos[:, i] = r_est - gt_pos[:, i]

        R_est = est_traj[i][0:3, 0:3]
        theta_vec = gt_rot[:, i]
        angle = np.linalg.norm(theta_vec)
        R_gt = np.eye(3) if angle < 1e-9 else axis_angle_to_rot(theta_vec)

        R_err = R_est @ R_gt.T
        tr = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        th = np.arccos(tr)
        if th < 1e-9:
            err_rot[:, i] = 0
        else:
            ln_R = (th / (2 * np.sin(th))) * (R_err - R_err.T)
            err_rot[:, i] = np.array([ln_R[2, 1], ln_R[0, 2], ln_R[1, 0]])

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"{title_suffix}")
    labels = ['x', 'y', 'z']

    for k in range(3):
        axs[k, 0].plot(times[:N], err_pos[k, :], 'b-', lw=1)
        axs[k, 0].plot(times[:N], 3 * sigma[k, :], 'r--', lw=1)
        axs[k, 0].plot(times[:N], -3 * sigma[k, :], 'r--', lw=1)
        axs[k, 0].set_ylabel(f'Pos {labels[k]} (m)')
        axs[k, 0].grid(True, alpha=0.5)

        axs[k, 1].plot(times[:N], err_rot[k, :], 'b-', lw=1)
        axs[k, 1].plot(times[:N], 3 * sigma[k + 3, :], 'r--', lw=1)
        axs[k, 1].plot(times[:N], -3 * sigma[k + 3, :], 'r--', lw=1)
        axs[k, 1].set_ylabel(f'Rot {labels[k]} (rad)')
        axs[k, 1].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()


def run_sliding_window(solver, k_start, k_end, window_size):
    print(f"\nProcessing Sliding Window (Window Size={window_size})...")
    total_steps = k_end - k_start + 1

    # 【关键修改 1】使用真正的弱先验
    # 原来的 inv(diag(1e-4)) = diag(10000) 太强了，会锁死第一帧导致漂移无法拉回
    # 这里用 1e-6 表示信息量极小，给予优化器调整窗口第一帧的自由度
    weak_prior = np.diag([1e-6] * 6)

    # 初始化第一帧的 Ground Truth
    theta0 = solver.gt_rot[:, k_start]
    R0 = axis_angle_to_rot(theta0) if np.linalg.norm(theta0) > 1e-9 else np.eye(3)
    T_init = np.eye(4)
    T_init[0:3, 0:3] = R0
    T_init[0:3, 3] = solver.gt_pos[:, k_start]

    results_T = []
    results_cov = np.zeros((6, total_steps))

    for t in range(total_steps):
        curr_k = k_start + t
        win_end_k = min(curr_k + window_size, k_end)

        # --- 1. 生成初始猜测 (Dead Reckoning) ---
        if t == 0:
            # 第一帧特殊处理：从 k_start 直接积分
            guess = solver.integrate_dead_reckoning(curr_k, win_end_k, T_init)
        else:
            # 【关键修改 2】解决 Off-by-one 索引错位问题
            # results_T[-1] 是 t-1 时刻 (即 curr_k - 1) 的最优结果
            # 我们必须从 (curr_k - 1) 开始积分，才能正确衔接 IMU 的这一步动态
            T_prev = results_T[-1]

            # 积分范围：[curr_k - 1, ..., win_end_k]
            full_guess = solver.integrate_dead_reckoning(curr_k - 1, win_end_k, T_prev)

            # 我们需要的优化变量是从 curr_k 开始的，所以丢掉第0个元素 (即 curr_k - 1)
            guess = full_guess[1:]

        # --- 2. 求解优化 ---
        # 始终使用 weak_prior，允许窗口随观测滑动
        opt_traj, _, full_cov_block = solver.solve(curr_k, win_end_k, guess, weak_prior, max_iter=5)

        # --- 3. 存储结果 ---
        # 只保存窗口第一帧的结果 (Marginalize out the rest implicitly)
        T_opt_curr = opt_traj[0]
        results_T.append(T_opt_curr)

        # 提取协方差对角线用于绘图
        cov_curr = full_cov_block[0:6, 0:6]
        results_cov[:, t] = np.diag(cov_curr)

        # 打印进度
        if t % 50 == 0:
            print(f"Step {t}/{total_steps}")

    return results_T, results_cov


def main():
    solver = BatchSolver(MAT_PATH)
    if not hasattr(solver, 'times'): return

    k1, k2 = 1214, 1713
    times = solver.times[k1: k2 + 1]

    # Q4 Plot
    print("Generating Q4 Plot...")
    valid_counts = []
    colors = []
    for k in range(k1, k2 + 1):
        valid = np.sum(solver.measurements[0, k, :] != -1)
        valid_counts.append(valid)
        colors.append('g' if valid >= 3 else 'r')
    plt.figure(figsize=(12, 4))
    plt.scatter(times, valid_counts, c=colors, s=10)
    plt.title("Landmark Visibility")
    plt.show()

    # Q5a
    print("\nRunning Q5a (Batch)...")
    theta0 = solver.gt_rot[:, k1]
    R0 = axis_angle_to_rot(theta0)
    T_init = np.eye(4);
    T_init[0:3, 0:3] = R0;
    T_init[0:3, 3] = solver.gt_pos[:, k1]
    batch_guess = solver.integrate_dead_reckoning(k1, k2, T_init)
    batch_prior = np.linalg.inv(np.diag([1e-4] * 6))

    # 修复点：接收3个值
    T_batch, cov_batch, _ = solver.solve(k1, k2, batch_guess, batch_prior, max_iter=15)
    visualize_results(times, solver.gt_pos[:, k1:k2 + 1], solver.gt_rot[:, k1:k2 + 1], T_batch, cov_batch, "Q5a Batch")

    # Q5b
    T_sw50, cov_sw50 = run_sliding_window(solver, k1, k2, 50)
    visualize_results(times, solver.gt_pos[:, k1:k2 + 1], solver.gt_rot[:, k1:k2 + 1], T_sw50, cov_sw50,
                      "Q5b Sliding Window 50")

    # Q5c
    T_sw10, cov_sw10 = run_sliding_window(solver, k1, k2, 10)
    visualize_results(times, solver.gt_pos[:, k1:k2 + 1], solver.gt_rot[:, k1:k2 + 1], T_sw10, cov_sw10,
                      "Q5c Sliding Window 10")


if __name__ == "__main__":
    main()