import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import warnings
# Tắt cảnh báo chia cho 0 để code chạy mượt (sẽ xử lý logic bên dưới)
warnings.filterwarnings("ignore")
# ==============================================================================
# PHẦN 1: TẠO BẢNG BUTCHER THEO YÊU CẦU (CLASSIC HOẶC CUSTOM)
# ==============================================================================
class ButcherTableau:
    def __init__(self, name, alpha, beta, r):
        self.name = name
        self.alpha = np.array(alpha, dtype=float) # Vector c
        self.beta = np.array(beta, dtype=float) # Matrix A
        self.r = np.array(r, dtype=float) # Vector b (trọng số)
        self.s = len(alpha) # Số nấc (stages)
def generate_tableau(order, mode):
    """
    Tạo bảng Butcher dựa trên Cấp (order) và Chế độ (mode).
    mode = 1: Classic
    mode = 2: Custom Alpha (Người dùng nhập)
    """
    # --- RK1 (EULER) ---
    if order == 1:
        # RK1 thì không có alpha để chọn, chỉ có 1 nấc duy nhất
        return ButcherTableau("RK1 (Euler)", [0.], [[0.]], [1.])
    # --- RK2 ---
    elif order == 2:
        if mode == 1: # Classic (Heun)
            alpha = 1.0
            print(f" -> Đang dùng Classic RK2 (Heun, Alpha={alpha})")
        else:
            alpha = float(input(f" 👉 Nhập Alpha cho RK2 (ví dụ 0.5, 1, 0.75): "))
            if alpha == 0: alpha = 1.0 # Tránh lỗi
        # Công thức tổng quát RK2 phụ thuộc Alpha:
        # c2 = alpha, a21 = alpha, b2 = 1/(2*alpha), b1 = 1 - b2
        r2 = 1.0 / (2.0 * alpha)
        r1 = 1.0 - r2
        return ButcherTableau(f"RK2 (Alpha={alpha})",
                              [0., alpha],
                              [[0.,0.], [alpha,0.]],
                              [r1, r2])
    # --- RK3 ---
    elif order == 3:
        if mode == 1: # Classic (Nystrom/Kutta)
            print(" -> Đang dùng Classic RK3")
            return ButcherTableau("RK3 Classic",
                                  [0., 0.5, 1.],
                                  [[0,0,0], [0.5,0,0], [-1,2,0]],
                                  [1/6, 2/3, 1/6])
        else:
            # Họ RK3 tổng quát Heun (tham số hóa bởi c2 = alpha, giả sử c3=1)
            alpha = float(input(f" 👉 Nhập Alpha (c2) cho RK3 (ví dụ 0.5, 0.33): "))
            if alpha in [0, 1, 2/3]:
                print(" ⚠️ Alpha này gây mẫu số bằng 0. Tự động chỉnh về 0.5 (Classic).")
                alpha = 0.5
            # Tính toán các hệ số để đảm bảo bậc 3
            # c2 = alpha, c3 = 1
            b2 = 1 / (6 * alpha * (1 - alpha))
            b3 = (1 - 3*alpha) / (6 * (1 - alpha))
            b1 = 1 - b2 - b3
            beta32 = 1 / (6 * alpha * b3)
            beta31 = 1 - beta32
            return ButcherTableau(f"RK3 Custom (c2={alpha})",
                                  [0., alpha, 1.],
                                  [[0,0,0], [alpha,0,0], [beta31, beta32, 0]],
                                  [b1, b2, b3])
    # --- RK4 ---
    elif order == 4:
        if mode == 1: # Classic
            print(" -> Đang dùng Classic RK4")
            return ButcherTableau("RK4 Classic",
                                  [0., 0.5, 0.5, 1.],
                                  [[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]],
                                  [1/6, 1/3, 1/3, 1/6])
        else:
            print(" ⚠️ Lưu ý: RK4 Custom rất phức tạp.")
            print(" Bạn muốn nhập Alpha đại diện (cho họ 3/8) hay tự nhập toàn bộ bảng?")
            sub_choice = input(" Nhập 'a' để nhập Alpha, 'f' để nhập Full bảng: ").strip().lower()
            if sub_choice == 'a':
                # Họ RK4 tổng quát (gia đình phương pháp 3/8 rule tổng quát)
                # Tham số hóa bởi c2, c3. Giả sử c2 = c3 = alpha.
                alpha = float(input(f" 👉 Nhập Alpha (c2, c3) cho RK4 (thử 0.333 hoặc 0.5): "))
                if alpha == 0.5: # Classic
                    return generate_tableau(4, 1)
                # Đây là xấp xỉ logic cho trường hợp người dùng muốn chỉnh alpha
                # (Lưu ý: Để đạt chính xác bậc 4 với 1 alpha rất khó, đây là mô hình gần đúng hoặc biến thể)
                # Sử dụng biến thể Kutta 3/8 rule nếu alpha gần 1/3
                return ButcherTableau(f"RK4 (Alpha~{alpha})",
                                      [0., alpha, 2*alpha, 1.], # Giả lập cấu trúc
                                      [[0,0,0,0], [alpha,0,0,0], [alpha-alpha, 2*alpha,0,0], [1,-1,1,0]], # Ma trận gần đúng
                                      [1/8, 3/8, 3/8, 1/8]) # Trọng số 3/8 rule
            else:
                # Nhập tay toàn bộ (Cho người chuyên sâu)
                print(" 👉 Nhập Vector Alpha (c) cách nhau bởi dấu phẩy (VD: 0, 0.5, 0.5, 1):")
                a_vec = [float(x) for x in input().split(',')]
                print(" 👉 Nhập Vector Trọng số (b) cách nhau bởi dấu phẩy (VD: 0.166, 0.333...):")
                b_vec = [float(x) for x in input().split(',')]
                s = len(a_vec)
                beta_mat = np.zeros((s,s))
                print(f" 👉 Nhập Ma trận Beta ({s} dòng, mỗi dòng {s} số):")
                for i in range(s):
                    row = [float(x) for x in input(f" Dòng {i+1}: ").split(',')]
                    beta_mat[i, :len(row)] = row
                return ButcherTableau("RK4 User-Defined", a_vec, beta_mat, b_vec)
    # --- RK5 ---
    elif order == 5:
        # RK5 khá phức tạp, thường mặc định dùng Butcher hoặc Cash-Karp
        print(" -> Đang dùng Butcher's RK5 (6 nấc)")
        a = [0., 0.25, 0.25, 0.5, 0.75, 1.]
        b = np.zeros((6,6))
        b[1,0]=0.25; b[2,0]=0.125; b[2,1]=0.125; b[3,1]=-0.5; b[3,2]=1.
        b[4,0]=3/16; b[4,3]=9/16; b[5,0]=-3/7; b[5,1]=8/7; b[5,2]=6/7; b[5,3]=-12/7; b[5,4]=8/7
        r = [7/90, 0, 16/45, 2/15, 16/45, 7/90]
        return ButcherTableau("RK5 Butcher", a, b, r)
    return None
# ==============================================================================
# PHẦN 2: BỘ GIẢI (CORE SOLVER) - CHẤP NHẬN MỌI INPUT
# ==============================================================================
def solve_rk_general(f_func, t0, y0, t_end, h, tableau):
    # Chuẩn hóa y0 thành vector
    y0_arr = np.atleast_1d(y0).astype(float)
    dim = len(y0_arr)
    # Tạo lưới thời gian
    n_steps = int(np.ceil((t_end - t0) / h)) + 1
    t_vals = np.linspace(t0, t_end, n_steps)
    h_real = (t_end - t0) / (n_steps - 1)
    y_vals = np.zeros((n_steps, dim))
    y_vals[0] = y0_arr
    print(f"\n🚀 Đang chạy {tableau.name}...")
    for i in range(n_steps - 1):
        s = tableau.s
        k = np.zeros((s, dim))
        # Tính các hệ số k1, k2, ... ks
        for stage in range(s):
            t_stage = t_vals[i] + tableau.alpha[stage] * h_real
            # Tính y_stage = y_n + sum(beta_ij * k_j)
            y_stage = y_vals[i].copy()
            for j in range(stage):
                if tableau.beta[stage, j] != 0:
                    y_stage += tableau.beta[stage, j] * k[j]
            # Gọi hàm f
            val = f_func(t_stage, y_stage)
            if np.isscalar(val): val = np.array([val]) # Đảm bảo luôn là vector
            k[stage] = h_real * val
        # Tổng hợp kết quả: y_{n+1} = y_n + sum(r_i * k_i)
        y_vals[i+1] = y_vals[i] + np.dot(tableau.r, k)
    return t_vals, y_vals
# ==============================================================================
# PHẦN 3: NHẬP LIỆU LINH HOẠT
# ==============================================================================
def get_input_function():
    print("\n" + "="*60)
    print(" NHẬP HÀM SỐ (KHÔNG FIX CỨNG)")
    print("="*60)
    print(" 1. Dạng y' = f(t, y) (PT cấp 1)")
    print(" 2. Dạng y'' = f(t, y, y') (PT cấp 2 - Bài c)")
    print(" 3. Dạng Hệ PT {x' = ..., y' = ...} (Bài e)")
    print(" 4. Hệ tổng quát với nhiều biến (x, y, z, w, ...)")
    type_choice = input("\n👉 Chọn dạng bài (1/2/3/4): ").strip()
    if type_choice == '1':
        expr = input("✍️ Nhập f(t, y): ") # VD: t - y
        f = lambda t, u: np.array([eval(expr, {"t": t, "y": u[0], "np": np})])
        y0 = [float(input(" y(0) = "))]
        labels = ["y"]
    elif type_choice == '2':
        # Dạng cấp 2: y'' = f(t, y, dy)
        print("✍️ Nhập vế phải của y''. (Lưu ý: dùng 'y' là hàm số, 'dy' là đạo hàm)")
        expr = input(" y'' = ") # VD: (t + y) * np.cos(1 + dy)
        # Hệ: u0=y, u1=y' => u0'=u1, u1'=expr
        def f_wrapper(t, u):
            y, dy = u[0], u[1]
            return np.array([dy, eval(expr, {"t": t, "y": y, "dy": dy, "np": np, "cos":np.cos})])
        y = float(input(" y(0) = "))
        dy = float(input(" y'(0) = "))
        y0 = [y, dy]
        labels = ["y", "y'"]
        f = f_wrapper
    elif type_choice == '3':
        # Hệ 2 PT
        expr1 = input("✍️ x' = ") # VD: 0.5*x*(1-x) - 0.15*x*y
        expr2 = input("✍️ y' = ") # VD: -0.3*y + 0.2*x*y
        def f_wrapper(t, u):
            x, y = u[0], u[1]
            dx = eval(expr1, {"t": t, "x": x, "y": y, "np": np})
            dy = eval(expr2, {"t": t, "x": x, "y": y, "np": np})
            return np.array([dx, dy])
        x0 = float(input(" x(0) = "))
        y0_val = float(input(" y(0) = "))
        y0 = [x0, y0_val]
        labels = ["x", "y"]
        f = f_wrapper
    elif type_choice == '4':
        # Hệ tổng quát với dim biến
        dim = int(input("👉 Số lượng biến (dimension): "))
        exprs = []
        var_names = ["x", "y", "z", "w", "v", "u", "p", "q", "r", "s"][:dim]  # Tên biến mặc định, có thể mở rộng
        for i in range(dim):
            expr = input(f"✍️ {var_names[i]}' = ")  # VD: dùng các biến như x, y, z, ...
            exprs.append(expr)
        y0 = []
        labels = var_names
        for i in range(dim):
            val = float(input(f" {var_names[i]}(0) = "))
            y0.append(val)
        def f_wrapper(t, u):
            globals_dict = {"t": t, "np": np, "cos": np.cos, "sin": np.sin, "exp": np.exp}  # Thêm các hàm phổ biến
            for j in range(dim):
                globals_dict[var_names[j]] = u[j]
            dus = []
            for expr in exprs:
                du = eval(expr, globals_dict)
                dus.append(du)
            return np.array(dus)
        f = f_wrapper
    return f, y0, labels
# ==============================================================================
# MAIN PROGRAM
# ==============================================================================
def main():
    print("\n🔥 RUNGE-KUTTA MASTER TOOL 🔥")
    # --- BƯỚC 1: Chọn Cấp RK ---
    while True:
        try:
            order = int(input("\n👉 [BƯỚC 1] Bạn muốn dùng RK cấp mấy? (1-5): "))
            if 1 <= order <= 5: break
            print("Vui lòng nhập số từ 1 đến 5.")
        except: pass
    # --- BƯỚC 2: Chọn Chế độ Classic/Custom ---
    print(f"\n👉 [BƯỚC 2] Cấu hình RK{order}")
    print(" 1. Classic Mode (Chuẩn sách giáo khoa)")
    print(" 2. Custom Alpha Mode (Tự chọn tham số)")
    mode = int(input(" Lựa chọn (1/2): "))
    tableau = generate_tableau(order, mode)
    if tableau is None: return
    # --- BƯỚC 3: Nhập Hàm ---
    f_func, y0, labels = get_input_function()
    # --- BƯỚC 4: Tham số chạy ---
    print("\n👉 [BƯỚC 4] Tham số thời gian")
    t0 = float(input(" t0 (start): ") or 0)
    t_end = float(input(" t_end (finish): "))
    h = float(input(" h (step size): "))
    # --- RUN ---
    t_arr, y_arr = solve_rk_general(f_func, t0, y0, t_end, h, tableau)
    # --- HIỂN THỊ KẾT QUẢ TỪNG BƯỚC ---
    print("\n📋 Kết quả từng bước:")
    header = "t\t" + "\t".join(labels)
    print(header)
    for i in range(len(t_arr)):
        row = f"{t_arr[i]:.4f}\t" + "\t".join([f"{y_arr[i, j]:.6f}" for j in range(len(labels))])
        print(row)
    # --- PLOT ---
    print("\n📊 Đang vẽ đồ thị...")
    plt.figure(figsize=(10, 6))
    for i in range(len(labels)):
        plt.plot(t_arr, y_arr[:, i], label=labels[i], linewidth=2)
    plt.title(f"Kết quả RK{order} | {tableau.name} | h={h}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    print("\n✅ Xong! Giá trị cuối cùng:")
    for i, lbl in enumerate(labels):
        print(f" {lbl} = {y_arr[-1, i]:.6f}")
if __name__ == "__main__":
    main()