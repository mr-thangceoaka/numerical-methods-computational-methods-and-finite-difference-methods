import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Cấu hình font tiếng Việt
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

class AdamsSolver:
    """
    Giải bài toán Cauchy (IVP) bằng phương pháp Adams-Bashforth và Adams-Moulton
    Hỗ trợ hệ phương trình vi phân bậc cao và đa chiều
    """

    def __init__(self, order, dimension):
        """
        Khởi tạo solver

        Parameters:
        -----------
        order : int
            Bậc của phương trình vi phân (1-5)
        dimension : int
            Số chiều của hệ phương trình (1-5)
        """
        self.order = order
        self.dimension = dimension
        self.f = None
        self.t0 = None
        self.y0 = None
        self.t_end = None
        self.h = None

    def set_problem(self, f, t0, y0, t_end, h):
        """
        Thiết lập bài toán

        Parameters:
        -----------
        f : function
            Hàm vế phải f(t, y) với y là vector
        t0 : float
            Thời điểm ban đầu
        y0 : array-like
            Vector điều kiện ban đầu (chiều = order * dimension)
        t_end : float
            Thời điểm kết thúc
        h : float
            Bước nhảy
        """
        self.f = f
        self.t0 = t0
        self.y0 = np.array(y0, dtype=float).flatten()
        self.t_end = t_end
        self.h = h
        self.n_steps = int((t_end - t0) / h)

        # Kiểm tra kích thước
        expected_size = self.order * self.dimension
        if len(self.y0) != expected_size:
            raise ValueError(f"y0 phải có {expected_size} phần tử (order={self.order}, dim={self.dimension})")

    def runge_kutta_4(self, n_initial):
        """
        Phương pháp Runge-Kutta bậc 4 để khởi động
        """
        t_init = np.zeros(n_initial)
        y_init = np.zeros((n_initial, len(self.y0)))

        t_init[0] = self.t0
        y_init[0] = self.y0

        print(f"\n📊 Khởi động bằng Runge-Kutta 4:")
        print(f"  t[0] = {t_init[0]:.6f}, y[0] = {y_init[0]}")

        for i in range(n_initial - 1):
            t_n = t_init[i]
            y_n = y_init[i]

            k1 = self.h * self.f(t_n, y_n)
            k2 = self.h * self.f(t_n + self.h/2, y_n + k1/2)
            k3 = self.h * self.f(t_n + self.h/2, y_n + k2/2)
            k4 = self.h * self.f(t_n + self.h, y_n + k3)

            y_init[i+1] = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6
            t_init[i+1] = t_n + self.h

            print(f"  t[{i+1}] = {t_init[i+1]:.6f}, y[{i+1}] = {y_init[i+1]}")

        return t_init, y_init

    def adams_bashforth_coefficients(self, s):
        """Hệ số Adams-Bashforth s-bước (tính toán động cho s bất kỳ)"""
        # Bảng hệ số có sẵn cho s nhỏ (tối ưu)
        coeffs = {
            1: [1],
            2: [3/2, -1/2],
            3: [23/12, -16/12, 5/12],
            4: [55/24, -59/24, 37/24, -9/24],
            5: [1901/720, -2774/720, 2616/720, -1274/720, 251/720],
            6: [4277/1440, -7923/1440, 9982/1440, -7298/1440, 2877/1440, -475/1440],
            7: [198721/60480, -447288/60480, 705549/60480, -688256/60480, 407139/60480, -134472/60480, 19087/60480],
            8: [434241/120960, -1152169/120960, 2183877/120960, -2664477/120960, 2102243/120960, -1041723/120960, 295767/120960, -36799/120960]
        }

        if s in coeffs:
            return np.array(coeffs[s])
        else:
            # Tính toán động bằng công thức tổng quát (cho s > 8)
            print(f"⚠️ Tính toán hệ số cho s={s} (có thể mất vài giây...)")
            return self._compute_adams_coefficients(s, 'bashforth')

    def adams_moulton_coefficients(self, s):
        """Hệ số Adams-Moulton s-bước (tính toán động cho s bất kỳ)"""
        # Bảng hệ số có sẵn cho s nhỏ (tối ưu)
        coeffs = {
            1: [1/2, 1/2],
            2: [5/12, 8/12, -1/12],
            3: [9/24, 19/24, -5/24, 1/24],
            4: [251/720, 646/720, -264/720, 106/720, -19/720],
            5: [475/1440, 1427/1440, -798/1440, 482/1440, -173/1440, 27/1440],
            6: [19087/60480, 65112/60480, -46461/60480, 37504/60480, -20211/60480, 6312/60480, -863/60480],
            7: [36799/120960, 139849/120960, -121797/120960, 123133/120960, -88547/120960, 41499/120960, -11351/120960, 1375/120960]
        }

        if s in coeffs:
            return np.array(coeffs[s])
        else:
            # Tính toán động bằng công thức tổng quát
            print(f"⚠️ Tính toán hệ số cho s={s} (có thể mất vài giây...)")
            return self._compute_adams_coefficients(s, 'moulton')

    def _compute_adams_coefficients(self, s, method_type):
        """
        Tính hệ số Adams bằng phương pháp sai phân Newton
        (cho s > giá trị có sẵn trong bảng)
        """
        from scipy.special import comb

        if method_type == 'bashforth':
            # Adams-Bashforth: tích phân từ t_n đến t_{n+1}
            # sử dụng đa thức nội suy qua s điểm: t_n, t_{n-1}, ..., t_{n-s+1}
            beta = np.zeros(s)
            for j in range(s):
                coeff = 0.0
                for i in range(j + 1):
                    sign = (-1)**i
                    binom = comb(j, i, exact=True)
                    integral_val = 1.0 / (i + 1)
                    coeff += sign * binom * integral_val
                beta[j] = coeff
            return beta
        else:  # moulton
            # Adams-Moulton: tích phân từ t_n đến t_{n+1}
            # sử dụng đa thức nội suy qua s+1 điểm: t_{n+1}, t_n, ..., t_{n-s+1}
            beta = np.zeros(s + 1)
            for j in range(s + 1):
                coeff = 0.0
                for i in range(j + 1):
                    sign = (-1)**i
                    binom = comb(j, i, exact=True)
                    if i == 0:
                        integral_val = 1.0
                    else:
                        integral_val = sum([(-1)**(k+1) / k for k in range(1, i + 1)])
                    coeff += sign * binom * integral_val
                beta[j] = coeff
            return beta

    def solve_adams_bashforth(self, s):
        """
        Giải bằng Adams-Bashforth s-bước
        """
        print(f"\n{'='*70}")
        print(f"🔵 ADAMS-BASHFORTH {s}-BƯỚC (Công thức HIỆN - β₀ = 0)")
        print(f"{'='*70}")

        # Khởi động
        t, y = self.runge_kutta_4(s)

        # Hệ số
        beta = self.adams_bashforth_coefficients(s)
        print(f"\n📐 Hệ số β = {beta}")

        # Tính các f ban đầu
        f_values = [self.f(t[i], y[i]) for i in range(s)]

        # Tiếp tục tính
        print(f"\n🔄 Bắt đầu tính toán từ bước {s}...")
        for n in range(s, self.n_steps + 1):
            sum_term = np.zeros(len(self.y0))
            for i in range(s):
                sum_term += beta[i] * f_values[-(i+1)]

            y_new = y[-1] + self.h * sum_term
            t_new = t[-1] + self.h

            t = np.append(t, t_new)
            y = np.vstack([y, y_new])
            f_values.append(self.f(t_new, y_new))

            # Hiển thị tiến trình
            if (n - s + 1) % max(1, (self.n_steps - s + 1) // 10) == 0:
                progress = (n - s + 1) / (self.n_steps - s + 1) * 100
                print(f"  Tiến trình: {progress:.1f}% - t = {t_new:.2f}")

        return t, y, 'Adams-Bashforth'

    def solve_adams_moulton(self, s, max_iter=20, tol=1e-10):
        """
        Giải bằng Adams-Moulton s-bước (ẨN)
        """
        print(f"\n{'='*70}")
        print(f"🟢 ADAMS-MOULTON {s}-BƯỚC (Công thức ẨN - β₀ ≠ 0)")
        print(f"{'='*70}")

        # Khởi động
        t, y = self.runge_kutta_4(s)

        # Hệ số
        beta = self.adams_moulton_coefficients(s)
        print(f"\n📐 Hệ số β = {beta}")
        print(f"   (β₀ = {beta[0]} ≠ 0 → Công thức ẨN)")

        # Tính các f ban đầu
        f_values = [self.f(t[i], y[i]) for i in range(s)]

        # Hệ số Adams-Bashforth để dự đoán
        beta_ab = self.adams_bashforth_coefficients(s)

        print(f"\n🔄 Bắt đầu tính toán từ bước {s}...")
        for n in range(s, self.n_steps + 1):
            t_new = t[-1] + self.h

            # Dự đoán bằng Adams-Bashforth
            sum_ab = np.zeros(len(self.y0))
            for i in range(s):
                sum_ab += beta_ab[i] * f_values[-(i+1)]
            y_predict = y[-1] + self.h * sum_ab

            # Lặp điểm bất động để hiệu chỉnh
            y_new = y_predict.copy()
            for iteration in range(max_iter):
                y_old = y_new.copy()

                sum_term = beta[0] * self.f(t_new, y_new)
                for i in range(1, len(beta)):
                    sum_term += beta[i] * f_values[-(i)]

                y_new = y[-1] + self.h * sum_term

                if np.linalg.norm(y_new - y_old) < tol:
                    break

            t = np.append(t, t_new)
            y = np.vstack([y, y_new])
            f_values.append(self.f(t_new, y_new))

            # Hiển thị tiến trình
            if (n - s + 1) % max(1, (self.n_steps - s + 1) // 10) == 0:
                progress = (n - s + 1) / (self.n_steps - s + 1) * 100
                print(f"  Tiến trình: {progress:.1f}% - t = {t_new:.2f}")

        return t, y, 'Adams-Moulton'

    def plot_solution(self, results_list):
        """
        Vẽ đồ thị nghiệm

        Parameters:
        -----------
        results_list : list of tuples
            [(t1, y1, name1), (t2, y2, name2), ...]
        """
        n_vars = len(self.y0)

        if n_vars == 1:
            # Đơn chiều - 1 đồ thị
            plt.figure(figsize=(12, 6))

            colors = ['blue', 'green', 'red', 'orange', 'purple']
            markers = ['o', 's', '^', 'D', 'v']

            for idx, (t, y, name) in enumerate(results_list):
                plt.plot(t, y, color=colors[idx % 5], marker=markers[idx % 5],
                         markersize=3, label=name, markevery=max(1, len(t)//50))

            plt.xlabel('t', fontsize=12)
            plt.ylabel('y', fontsize=12)
            plt.title(f'Nghiệm bài toán bậc {self.order}, chiều {self.dimension}',
                      fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

        else:
            # Đa chiều - nhiều subplots
            n_cols = min(3, n_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            if n_vars == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

            colors = ['blue', 'green', 'red', 'orange', 'purple']
            markers = ['o', 's', '^', 'D', 'v']

            for var_idx in range(n_vars):
                ax = axes[var_idx]

                for res_idx, (t, y, name) in enumerate(results_list):
                    ax.plot(t, y[:, var_idx], color=colors[res_idx % 5],
                            marker=markers[res_idx % 5], markersize=3, label=name,
                            markevery=max(1, len(t)//50))

                ax.set_xlabel('t', fontsize=10)
                ax.set_ylabel(f'y[{var_idx}]', fontsize=10)
                ax.set_title(f'Thành phần y[{var_idx}]', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

            # Ẩn các subplot thừa
            for idx in range(n_vars, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.show()


def convert_high_order_to_system(order):
    """
    Chuyển phương trình bậc cao thành hệ bậc 1

    Ví dụ: y'' = f(t, y, y') -> z0 = y, z1 = y'
           Hệ: z0' = z1
                z1' = f(t, z0, z1)
    """
    print(f"\n💡 Lưu ý: Phương trình bậc {order} sẽ được chuyển thành hệ {order} phương trình bậc 1")
    print(f"   Đặt: z[0] = y, z[1] = y', z[2] = y'', ..., z[{order-1}] = y^({order-1})")
    print(f"   Khi nhập hàm f, sử dụng: z[0] thay cho y, z[1] thay cho y', v.v...")


def input_function(order, dimension, variable_name='t'):
    """
    Nhập hàm f cho bài toán
    """
    if order == 1:
        if dimension == 1:
            print(f"\nNhập hàm f({variable_name}, y):")
            print("Ví dụ: -2*y  hoặc  y**3 * np.sin({variable_name} + y)")
            f_str = input("f = ")
            return lambda t, y: np.array([eval(f_str.replace(variable_name, str(t)).replace('y', 'y[0]'))])
        else:
            print(f"\nNhập hệ {dimension} phương trình:")
            f_strs = []
            for i in range(dimension):
                print(f"  f{i+1}({variable_name}, y1, y2, ..., y{dimension}) = ", end='')
                print(f"(Dùng y[0], y[1], ..., y[{dimension-1}])")
                f_str = input(f"  f{i+1} = ")
                f_strs.append(f_str)

            def system_f(t, y):
                result = []
                for f_str in f_strs:
                    expr = f_str.replace(variable_name, str(t))
                    for j in range(dimension):
                        expr = expr.replace(f'y{j+1}', f'y[{j}]')
                    result.append(eval(expr))
                return np.array(result)

            return system_f
    else:
        convert_high_order_to_system(order)

        if dimension == 1:
            print(f"\nNhập vế phải của phương trình bậc {order}:")
            print(f"y^({order}) = f({variable_name}, y, y', ..., y^({order-1}))")
            print(f"Ví dụ: ({variable_name} + z[0]) * np.cos(1 + z[1])")
            f_str = input("f = ")

            def high_order_f(t, z):
                result = np.zeros(order)
                for i in range(order - 1):
                    result[i] = z[i + 1]
                expr = f_str.replace(variable_name, str(t))
                result[order - 1] = eval(expr)
                return result

            return high_order_f
        else:
            # Hệ bậc cao, đa chiều
            print(f"\nNhập hệ {dimension} phương trình bậc {order}:")
            print("Cần cung cấp tổng cộng {} phương trình bậc 1".format(order * dimension))

            f_strs = []
            for i in range(dimension):
                print(f"\n--- Phương trình cho biến thứ {i+1} ---")
                for j in range(order):
                    if j < order - 1:
                        print(f"  z{i}_{j}' = z{i}_{j+1}  (tự động)")
                    else:
                        print(f"  z{i}_{j}' = ", end='')
                        f_str = input()
                        f_strs.append(f_str)

            def system_high_order_f(t, z):
                result = np.zeros(order * dimension)
                idx = 0
                for i in range(dimension):
                    for j in range(order):
                        if j < order - 1:
                            result[idx] = z[idx + 1]
                        else:
                            expr = f_strs[i].replace(variable_name, str(t))
                            result[idx] = eval(expr)
                        idx += 1
                return result

            return system_high_order_f


def main():
    """Chương trình chính"""
    print("="*70)
    print(" "*15 + "GIẢI BÀI TOÁN CAUCHY (IVP)")
    print(" "*10 + "Adams-Bashforth & Adams-Moulton")
    print("="*70)

    # Bước 1: Chọn phương pháp
    print("\n" + "="*70)
    print("BƯỚC 1: CHỌN PHƯƠNG PHÁP")
    print("="*70)
    print("1. Adams-Bashforth (ABs) - Công thức HIỆN (β₀ = 0)")
    print("2. Adams-Moulton (AMs) - Công thức ẨN (β₀ ≠ 0)")
    print("3. So sánh cả hai phương pháp")

    method_choice = int(input("\nChọn phương pháp (1-3): "))

    # Bước 2: Chọn số bước
    print("\n" + "="*70)
    print("BƯỚC 2: CHỌN SỐ BƯỚC (s)")
    print("="*70)

    if method_choice in [1, 3]:
        print("Adams-Bashforth: s ∈ {1, 2, 3, 4, 5}")
        s_ab = int(input("Số bước s cho ABs: "))
        if not (1 <= s_ab <= 5):
            print("⚠️ s phải từ 1-5. Đặt s = 4")
            s_ab = 4

    if method_choice in [2, 3]:
        print("Adams-Moulton: s ∈ {1, 2, 3, 4, 5}")
        s_am = int(input("Số bước s cho AMs: "))
        if not (1 <= s_am <= 5):
            print("⚠️ s phải từ 1-5. Đặt s = 4")
            s_am = 4

    # Bước 3: Chọn bậc và số chiều
    print("\n" + "="*70)
    print("BƯỚC 3: ĐỊNH NGHĨA BÀI TOÁN")
    print("="*70)

    print("\nChọn loại bài toán:")
    print("1. Bài toán mẫu (6 bài toán có sẵn)")
    print("2. Nhập bài toán tùy chỉnh")

    problem_type = int(input("\nChọn (1-2): "))

    if problem_type == 1:
        # Bài toán mẫu
        print("\n--- BÀI TOÁN MẪU ---")
        print("a) y'(t) = -2y, y(0) = 1, t ∈ [0,100], h = 0.1")
        print("b) y'(t) = ty³sin(t+y), y(0) = -0.2, t ∈ [0,10], h = 0.1")
        print("c) y''(t) = (t+y)cos(1+y'), y(0)=1, y'(0)=-1, t ∈ [0,20], h = 0.1")
        print("d) y'''(t) = (1+ty')sin(1+yy')/(1+y²+(y'')²), y(0)=1, y'(0)=0.5, y''(0)=-1, t ∈ [0,10], h = 0.05")
        print("e) Hệ 2 PT: x' = 0.5x(1-x)-0.15xy, y' = -0.3y+0.2xy, t ∈ [0,2000], h = 0.1")
        print("f) Hệ 3 PT: Lotka-Volterra 3 loài, t ∈ [0,1500], h = 0.1")

        choice = input("\nChọn bài toán (a-f): ").lower()

        if choice == 'a':
            order, dimension = 1, 1
            f = lambda t, y: np.array([-2 * y[0]])
            t0, y0, t_end, h = 0, [1], 100, 0.1

        elif choice == 'b':
            order, dimension = 1, 1
            f = lambda t, y: np.array([t * y[0]**3 * np.sin(t + y[0])])
            t0, y0, t_end, h = 0, [-0.2], 10, 0.1

        elif choice == 'c':
            order, dimension = 2, 1
            def f(t, z):  # z[0] = y, z[1] = y'
                return np.array([z[1], (t + z[0]) * np.cos(1 + z[1])])
            t0, y0, t_end, h = 0, [1, -1], 20, 0.1

        elif choice == 'd':
            order, dimension = 3, 1
            def f(t, z):  # z[0] = y, z[1] = y', z[2] = y''
                numerator = (1 + t * z[1]) * np.sin(1 + z[0] * z[1])
                denominator = 1 + z[0]**2 + z[2]**2
                return np.array([z[1], z[2], numerator / denominator])
            t0, y0, t_end, h = 0, [1, 0.5, -1], 10, 0.05

        elif choice == 'e':
            order, dimension = 1, 2
            def f(t, y):  # y[0] = x, y[1] = y
                return np.array([
                    0.5 * y[0] * (1 - y[0]) - 0.15 * y[0] * y[1],
                    -0.3 * y[1] + 0.2 * y[0] * y[1]
                ])
            t0, y0, t_end, h = 0, [0.7, 0.5], 2000, 0.1

        elif choice == 'f':
            order, dimension = 1, 3
            def f(t, y):  # y[0] = x, y[1] = y, y[2] = z
                return np.array([
                    0.4 * y[0] * (1 - y[0]/20) + 0.4 * y[1] - 0.3 * y[0] * y[2],
                    0.7 * y[1] * (1 - y[1]/25) - 0.4 * y[1] - 0.4 * y[1] * y[2],
                    -0.3 * y[2] + 0.35 * (y[0] + y[1]) * y[2]
                ])
            t0, y0, t_end, h = 0, [12, 18, 8], 1500, 0.1

        else:
            print("Lựa chọn không hợp lệ!")
            return

    else:
        # Nhập tùy chỉnh
        order = int(input("\nBậc của phương trình (1-5): "))
        dimension = int(input("Số chiều của hệ (1-5): "))

        if not (1 <= order <= 5 and 1 <= dimension <= 5):
            print("⚠️ Bậc và chiều phải từ 1-5!")
            return

        f = input_function(order, dimension)

        t0 = float(input("\nThời điểm ban đầu t0: "))
        t_end = float(input("Thời điểm kết thúc t_end: "))
        h = float(input("Bước nhảy h: "))

        print(f"\nNhập điều kiện ban đầu ({order * dimension} giá trị):")
        if order == 1:
            if dimension == 1:
                y0_val = float(input("  y(t0) = "))
                y0 = [y0_val]
            else:
                y0 = []
                for i in range(dimension):
                    val = float(input(f"  y{i+1}(t0) = "))
                    y0.append(val)
        else:
            if dimension == 1:
                y0 = []
                for i in range(order):
                    val = float(input(f"  y^({i})(t0) = "))
                    y0.append(val)
            else:
                y0 = []
                for i in range(dimension):
                    for j in range(order):
                        val = float(input(f"  z{i}_{j}(t0) = "))
                        y0.append(val)

    # Khởi tạo solver
    solver = AdamsSolver(order, dimension)
    solver.set_problem(f, t0, y0, t_end, h)

    print("\n" + "="*70)
    print("THÔNG TIN BÀI TOÁN")
    print("="*70)
    print(f"Bậc: {order}, Chiều: {dimension}")
    print(f"Khoảng: [{t0}, {t_end}], Bước: h = {h}")
    print(f"Điều kiện ban đầu: y0 = {y0}")
    print(f"Số bước tính toán: {solver.n_steps}")

    # Giải bài toán
    results = []

    if method_choice == 1:
        t, y, name = solver.solve_adams_bashforth(s_ab)
        results.append((t, y, f'{name} {s_ab}-bước'))

    elif method_choice == 2:
        t, y, name = solver.solve_adams_moulton(s_am)
        results.append((t, y, f'{name} {s_am}-bước'))

    else:
        t_ab, y_ab, name_ab = solver.solve_adams_bashforth(s_ab)
        t_am, y_am, name_am = solver.solve_adams_moulton(s_am)
        results.append((t_ab, y_ab, f'{name_ab} {s_ab}-bước'))
        results.append((t_am, y_am, f'{name_am} {s_am}-bước'))

    # Hiển thị kết quả
    print("\n" + "="*70)
    print("KẾT QUẢ CUỐI CÙNG")
    print("="*70)
    for t, y, name in results:
        print(f"\n{name}:")
        print(f"  t = {t[-1]:.6f}")
        print(f"  y = {y[-1]}")

    # Vẽ đồ thị
    print("\n📊 Đang vẽ đồ thị...")
    solver.plot_solution(results)

    print("\n" + "="*70)
    print(" "*25 + "✅ HOÀN THÀNH!")
    print("="*70)


if __name__ == "__main__":
    main()