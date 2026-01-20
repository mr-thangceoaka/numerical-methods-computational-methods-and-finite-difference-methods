"""
CHƯƠNG TRÌNH GIẢI BÀI TOÁN CAUCHY TOÀN DIỆN
Tính năng: Euler hiện, Euler ẩn, Hình thang
Cập nhật: Hỗ trợ nhập liệu tự nhiên (x, y, z, t)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sys
import math

class EulerSolver:
    """Lớp giải bài toán Cauchy với giao diện biến tự nhiên"""

    def __init__(self):
        self.methods = {
            '1': 'Euler hiện (Euler Forward)',
            '2': 'Euler ẩn (Euler Backward)',
            '3': 'Hình thang (Trapezoidal)'
        }
        # Kho biến tên gọi cho hệ phương trình
        self.system_vars = ['x', 'y', 'z', 'u', 'v', 'w']

    def display_welcome(self):
        """Hiển thị màn hình chào"""
        print("=" * 80)
        print("CHƯƠNG TRÌNH GIẢI BÀI TOÁN CAUCHY (HỖ TRỢ BIẾN x, y, z)")
        print("=" * 80)
        print("Quy ước đặt tên biến:")
        print("1. Bài toán 1 chiều:  Biến độc lập là 'x', hàm cần tìm là 'y'")
        print("                       (Ví dụ: y' = x + y)")
        print("2. Hệ phương trình:   Biến độc lập là 't' (thời gian).")
        print("                       Các hàm cần tìm là 'x', 'y', 'z'...")
        print("                       (Ví dụ: dx/dt = y, dy/dt = -x)")
        print("-" * 80)
        print("Các phương pháp số:")
        for key, value in self.methods.items():
            print(f"  {key}. {value}")
        print()

    def get_problem_context(self):
        """Xác định loại bài toán và tên biến"""
        print("\n--- BƯỚC 1: CẤU HÌNH BÀI TOÁN ---")
        print("1. Phương trình vô hướng (1 chiều)")
        print("2. Hệ phương trình (n chiều)")

        while True:
            choice = input("Chọn loại (1/2): ").strip()
            if choice == '1':
                # Cấu hình cho 1 chiều
                return {
                    'type': 'scalar',
                    'dim': 1,
                    'indep_var': 'x',    # Biến độc lập
                    'dep_vars': ['y']    # Biến phụ thuộc
                }
            elif choice == '2':
                # Cấu hình cho hệ phương trình
                while True:
                    try:
                        dim = int(input("Nhập số chiều của hệ (số phương trình): "))
                        if dim > 0: break
                        print("Số chiều phải > 0.")
                    except ValueError:
                        print("Vui lòng nhập số nguyên.")

                # Tạo tên biến: x, y, z hoặc x1, x2...
                if dim <= len(self.system_vars):
                    names = self.system_vars[:dim]
                else:
                    names = [f"x{i+1}" for i in range(dim)]

                return {
                    'type': 'system',
                    'dim': dim,
                    'indep_var': 't',
                    'dep_vars': names
                }
            print("Lựa chọn không hợp lệ.")

    def get_function(self, context):
        """Lấy hàm f(t, y) dựa trên input người dùng"""
        print("\n--- BƯỚC 2: NHẬP HÀM SỐ ---")

        dim = context['dim']
        indep = context['indep_var']
        deps = context['dep_vars']

        # --- TRƯỜNG HỢP 1: VÔ HƯỚNG (1 CHIỀU) ---
        if context['type'] == 'scalar':
            print(f"Nhập biểu thức cho y' = f({indep}, y)")
            print("Các hàm mẫu:")
            print("  1. y' = -y")
            print("  2. y' = x + y")
            print("  3. Tùy chỉnh (Nhập biểu thức)")

            c = input("Chọn (1-3): ").strip()
            if c == '1': return lambda x, y: -y, "y' = -y"
            elif c == '2': return lambda x, y: x + y, "y' = x + y"

            # Nhập tùy chỉnh
            print(f"\nNhập biểu thức f({indep}, y). Ví dụ: {indep}**2 + y, np.sin({indep})*y")
            expr = input(f"f({indep}, y) = ")

            def scalar_f(x_val, y_val):
                # Mapping environment
                local_env = {**math.__dict__, 'np': np, 'x': x_val, 'y': y_val}
                return eval(expr, {"__builtins__": None}, local_env)

            return scalar_f, f"y' = {expr}"

        # --- TRƯỜNG HỢP 2: HỆ PHƯƠNG TRÌNH ---
        else:
            print(f"Hệ phương trình với biến thời gian '{indep}' và các hàm {deps}")
            print("Các hệ mẫu:")
            print("  1. Hệ thú mồi (Lotka-Volterra) [x, y]")
            print("  2. Dao động điều hòa [x, y]")
            print("  3. Tùy chỉnh")

            c = input("Chọn (1-3): ").strip()

            if c == '1' and dim == 2:
                # Lotka-Volterra hardcode cho nhanh
                r = float(input("Nhập r (sinh trưởng, vd 1.0): ") or "1.0")
                a = float(input("Nhập a (tương tác, vd 0.1): ") or "0.1")
                desc = "Thú mồi (x, y)"
                def lv_func(t, y_vec):
                    x, y = y_vec[0], y_vec[1]
                    return np.array([r*x - a*x*y, -0.5*y + 0.02*x*y])
                return lv_func, desc

            elif c == '2' and dim == 2:
                desc = "Dao động: x'=y, y'=-x"
                def osc_func(t, y_vec):
                    return np.array([y_vec[1], -y_vec[0]])
                return osc_func, desc

            else:
                # Nhập tùy chỉnh từng dòng
                print("\nNhập các biểu thức đạo hàm (sử dụng: t, x, y, z...):")
                expressions = []
                for var in deps:
                    expr = input(f"d{var}/dt = ")
                    expressions.append(expr)

                def system_f(t_val, y_vec):
                    # Tạo từ điển biến cục bộ
                    local_env = {**math.__dict__, 'np': np, 't': t_val}

                    # Map giá trị y_vec vào tên biến (x, y, z...)
                    for i, name in enumerate(deps):
                        local_env[name] = y_vec[i]

                    # Tính toán
                    res = []
                    for e in expressions:
                        try:
                            res.append(eval(e, {"__builtins__": None}, local_env))
                        except:
                            res.append(0.0)
                    return np.array(res)

                full_desc = ", ".join([f"{v}'={e}" for v, e in zip(deps, expressions)])
                return system_f, full_desc

    def get_parameters(self, context):
        """Nhập tham số x0, y0, h"""
        print("\n--- BƯỚC 3: THAM SỐ CHẠY ---")
        indep = context['indep_var']
        deps = context['dep_vars']

        # Nhập giá trị đầu của biến độc lập
        t0 = float(input(f"Nhập giá trị đầu {indep}₀: "))

        # Nhập giá trị đầu của biến phụ thuộc
        if context['type'] == 'scalar':
            y0 = float(input(f"Nhập giá trị đầu y({t0}): "))
        else:
            print(f"Nhập điều kiện ban đầu tại {indep} = {t0}:")
            y0_list = []
            for var in deps:
                val = float(input(f"  {var}({t0}) = "))
                y0_list.append(val)
            y0 = np.array(y0_list)

        t_end = float(input(f"Nhập giá trị cuối {indep}_end: "))
        h = float(input("Nhập bước nhảy h: "))

        print("\n--- CẤU HÌNH HIỂN THỊ ---")
        decimals = int(input("Số chữ số thập phân (mặc định 6): ") or "6")
        show_steps = input("Hiển thị từng bước? (y/n, mặc định y): ").strip().lower() != 'n'

        return t0, y0, t_end, h, decimals, show_steps

    def print_header(self, context, decimals):
        """In tiêu đề bảng kết quả"""
        indep = context['indep_var']
        deps = context['dep_vars']

        # Formatting width
        w = decimals + 6
        header = f"{'Bước':<6} | {indep:<{w}} | "

        if context['type'] == 'scalar':
            header += f"{'y':<{w}} | f({indep},y)"
        else:
            # Hệ phương trình: in x, y, z...
            vals_str = " | ".join([f"{v:<{w}}" for v in deps])
            header += f"{vals_str}"

        print("-" * len(header))
        print(header)
        print("-" * len(header))

    def print_step(self, step, t, y, context, decimals, f_val=None):
        """In một dòng kết quả"""
        indep = context['indep_var']
        deps = context['dep_vars']
        w = decimals + 6
        fmt = f"{{:.{decimals}f}}"

        line = f"{step:<6d} | {fmt.format(t):<{w}} | "

        if context['type'] == 'scalar':
            line += f"{fmt.format(y):<{w}}"
            if f_val is not None:
                # f_val có thể là mảng 1 phần tử hoặc số
                val = f_val if np.isscalar(f_val) else f_val[0]
                line += f" | {fmt.format(val)}"
        else:
            # In các giá trị x, y, z...
            vals = [fmt.format(val) for val in y]
            line += " | ".join([f"{v:<{w}}" for v in vals])

        print(line)

    # ==========================================
    # LOGIC TOÁN HỌC (CORE SOLVERS)
    # ==========================================

    def run_solver(self, method_choice, f, t0, y0, t_end, h, context, decimals, show_steps):
        """Hàm điều khiển chung cho việc giải"""

        t_values = [t0]
        y_values = [y0] # List chứa các giá trị y (scalar hoặc array)

        t = t0
        y = y0 if np.isscalar(y0) else np.array(y0)
        is_scalar = context['type'] == 'scalar'

        if show_steps:
            print(f"\nKẾT QUẢ CHI TIẾT ({self.methods[method_choice]})")
            self.print_header(context, decimals)
            self.print_step(0, t, y, context, decimals)

        step = 1

        while t < t_end - h/10: # Trừ epsilon để tránh lỗi làm tròn

            # --- 1. EULER HIỆN ---
            if method_choice == '1':
                f_curr = f(t, y) if is_scalar else f(t, y)
                # Xử lý kết quả f trả về nếu là scalar function
                if is_scalar and isinstance(f_curr, np.ndarray): f_curr = f_curr[0]

                y_new = y + h * f_curr
                t_new = t + h

            # --- 2. EULER ẨN ---
            elif method_choice == '2':
                t_new = t + h

                # Hàm cần tìm nghiệm: Z - y_curr - h*f(t_new, Z) = 0
                def eq_backward(y_next):
                    if is_scalar:
                        # y_next là array 1 phần tử do fsolve truyền vào
                        val = y_next[0]
                        res = val - y - h * f(t_new, val)
                        return res if np.isscalar(res) else res[0]
                    else:
                        return y_next - y - h * f(t_new, y_next)

                # Dự báo ban đầu (Euler hiện)
                guess = y + h * (f(t, y) if not is_scalar else f(t, y))

                try:
                    y_root = fsolve(eq_backward, guess)
                    y_new = y_root[0] if is_scalar else y_root
                except:
                    print(f"\nLỗi hội tụ tại bước {step}")
                    break

            # --- 3. HÌNH THANG ---
            elif method_choice == '3':
                t_new = t + h
                f_curr = f(t, y)
                if is_scalar and isinstance(f_curr, np.ndarray): f_curr = f_curr[0]

                # Hàm: Z - y - h/2 * (f_curr + f(t_new, Z)) = 0
                def eq_trap(y_next):
                    if is_scalar:
                        val = y_next[0]
                        f_next = f(t_new, val)
                        if isinstance(f_next, np.ndarray): f_next = f_next[0]
                        return val - y - (h/2)*(f_curr + f_next)
                    else:
                        return y_next - y - (h/2)*(f_curr + f(t_new, y_next))

                guess = y + h * f_curr
                try:
                    y_root = fsolve(eq_trap, guess)
                    y_new = y_root[0] if is_scalar else y_root
                except:
                    print(f"\nLỗi hội tụ tại bước {step}")
                    break

            # Cập nhật và lưu trữ
            t_values.append(t_new)
            # Copy nếu là array để tránh tham chiếu
            y_store = y_new if is_scalar else y_new.copy()
            y_values.append(y_store)

            if show_steps:
                # Tính f tại điểm mới để hiển thị (chỉ mang tính tham khảo)
                f_disp = f(t_new, y_new)
                self.print_step(step, t_new, y_new, context, decimals, f_disp)

            t, y = t_new, y_new
            step += 1

        return np.array(t_values), np.array(y_values)

    def plot_results(self, t_vals, y_vals, context, method_name, desc):
        """Vẽ đồ thị kết quả"""
        print("\nĐang vẽ đồ thị...")
        plt.figure(figsize=(12, 6))

        indep = context['indep_var']
        deps = context['dep_vars']

        # Plot 1: Các thành phần theo thời gian
        plt.subplot(1, 2, 1)
        if context['type'] == 'scalar':
            plt.plot(t_vals, y_vals, 'b-o', label=deps[0], markersize=3)
        else:
            for i, name in enumerate(deps):
                plt.plot(t_vals, y_vals[:, i], '-o', label=name, markersize=3)

        plt.title(f"Đồ thị theo {indep}\n{desc}")
        plt.xlabel(indep)
        plt.ylabel("Giá trị")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot 2: Quỹ đạo pha (nếu là hệ phương trình)
        if context['type'] == 'system':
            plt.subplot(1, 2, 2)
            if context['dim'] >= 2:
                # Vẽ biến 1 vs biến 2 (ví dụ x vs y)
                x_idx, y_idx = 0, 1
                name_x, name_y = deps[x_idx], deps[y_idx]

                plt.plot(y_vals[:, x_idx], y_vals[:, y_idx], 'r-')
                plt.plot(y_vals[0, x_idx], y_vals[0, y_idx], 'go', label='Bắt đầu')
                plt.plot(y_vals[-1, x_idx], y_vals[-1, y_idx], 'ks', label='Kết thúc')

                plt.title(f"Quỹ đạo pha ({name_x} vs {name_y})")
                plt.xlabel(name_x)
                plt.ylabel(name_y)
                plt.grid(True, alpha=0.3)
                plt.legend()
            else:
                plt.text(0.5, 0.5, "Cần ít nhất 2 biến để vẽ pha", ha='center')
        else:
            # Nếu là 1 chiều, vẽ lại nhưng zoom vào hoặc style khác
            plt.subplot(1, 2, 2)
            plt.plot(t_vals, y_vals, 'g--')
            plt.title("Overview")
            plt.xlabel(indep)

        plt.tight_layout()
        plt.show()

    def run(self):
        """Hàm chính chạy chương trình"""
        self.display_welcome()

        # 1. Cấu hình
        context = self.get_problem_context()

        # 2. Nhập hàm
        f, desc = self.get_function(context)

        # 3. Chọn phương pháp
        print("\n--- CHỌN PHƯƠNG PHÁP ---")
        for k, v in self.methods.items(): print(f"{k}. {v}")
        m_choice = input("Chọn (1/2/3): ").strip()
        if m_choice not in self.methods:
            print("Lựa chọn sai. Mặc định dùng Euler hiện (1).")
            m_choice = '1'

        # 4. Nhập tham số
        t0, y0, t_end, h, dec, show = self.get_parameters(context)

        # 5. Chạy solver
        print("\n🔄 Đang tính toán...")
        try:
            ts, ys = self.run_solver(m_choice, f, t0, y0, t_end, h, context, dec, show)

            # 6. Kết luận
            print("\n" + "="*40)
            print("KẾT QUẢ CUỐI CÙNG")
            print(f"Tại {context['indep_var']} = {ts[-1]:.{dec}f}:")
            if context['type'] == 'scalar':
                print(f"{context['dep_vars'][0]} ≈ {ys[-1]:.{dec}f}")
            else:
                for i, name in enumerate(context['dep_vars']):
                    print(f"{name} ≈ {ys[-1][i]:.{dec}f}")
            print("="*40)

            # 7. Vẽ
            q = input("\nVẽ đồ thị? (y/n): ").lower()
            if q != 'n':
                self.plot_results(ts, ys, context, self.methods[m_choice], desc)

        except Exception as e:
            print(f"\n❌ Đã xảy ra lỗi trong quá trình tính toán: {e}")
            print("Gợi ý: Kiểm tra biểu thức hàm số hoặc giảm bước nhảy h.")

if __name__ == "__main__":
    solver = EulerSolver()
    try:
        solver.run()
    except KeyboardInterrupt:
        print("\n\nĐã dừng chương trình.")