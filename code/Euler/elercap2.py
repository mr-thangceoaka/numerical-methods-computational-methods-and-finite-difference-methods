"""
CHƯƠNG TRÌNH GIẢI BÀI TOÁN CAUCHY - CHẾ ĐỘ GIẢNG GIẢI (EXPLAINER MODE)
Tính năng:
- Giải phương trình đa bậc.
- Hiển thị công thức và quá trình thay số từng bước.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sys
import math

class EulerSolverExplainer:
    def __init__(self):
        self.methods = {
            '1': 'Euler hiện (Explicit Euler)',
            '2': 'Euler ẩn (Implicit Euler)',
            '3': 'Hình thang (Trapezoidal)'
        }

    def display_welcome(self):
        print("=" * 90)
        print("GIẢI PHƯƠNG TRÌNH VI PHÂN - HIỂN THỊ CÔNG THỨC TOÁN HỌC")
        print("=" * 90)

    def get_problem_context(self):
        # ... (Giữ nguyên logic nhập liệu như phiên bản trước) ...
        # Để tiết kiệm không gian, tôi tóm tắt phần này, logic không đổi
        while True:
            print("\n--- BƯỚC 1: CHỌN LOẠI BÀI TOÁN ---")
            print("1. Phương trình cấp 1 (Scalar) -> y' = f(x,y)")
            print("2. Hệ phương trình (System) -> x' = ..., y' = ...")
            print("3. Phương trình cấp cao (High Order) -> y'', y'''... (Tự động hạ bậc)")

            choice = input("Lựa chọn (1/2/3): ").strip()

            if choice == '1':
                return {'type': 'scalar', 'dim': 1, 'indep': 'x', 'vars': ['y']}
            elif choice == '2':
                try:
                    dim = int(input("Nhập số phương trình: "))
                    vars_list = ['x', 'y', 'z', 'u', 'v'][:dim] if dim <= 5 else [f"x{i+1}" for i in range(dim)]
                    return {'type': 'system', 'dim': dim, 'indep': 't', 'vars': vars_list}
                except: continue
            elif choice == '3':
                try:
                    order = int(input("Nhập cấp phương trình (vd: 2, 3): "))
                    vars_list = ['y'] + [f"y'" if i==1 else f"y^({i})" for i in range(1, order)]
                    return {'type': 'high_order', 'dim': order, 'indep': 't', 'vars': vars_list, 'order': order}
                except: continue

    def get_function(self, context):
        # ... (Giữ nguyên logic nhập hàm f như phiên bản trước) ...
        print("\n--- BƯỚC 2: NHẬP BIỂU THỨC ---")
        indep = context['indep']

        if context['type'] == 'high_order':
            order = context['order']
            print(f"Nhập vế phải cho đạo hàm cao nhất y^({order}):")
            print("Quy ước biến: y, dy (y'), d2y (y'')...")
            expr = input(f"y^({order}) = ")

            def high_order_f(t_val, state_vec):
                local_env = {**math.__dict__, 'np': np, 't': t_val}
                local_env['y'] = state_vec[0]
                if order > 1: local_env['dy'] = state_vec[1]
                if order > 2:
                    for i in range(2, order): local_env[f"d{i}y"] = state_vec[i]
                try: val = eval(expr, {"__builtins__": None}, local_env)
                except: val = 0.0
                res = list(state_vec[1:])
                res.append(val)
                return np.array(res)
            return high_order_f, expr

        elif context['type'] == 'system':
            vars_list = context['vars']
            print(f"Nhập các biểu thức (dùng {indep}, {', '.join(vars_list)}):")
            exprs = []
            for v in vars_list:
                exprs.append(input(f"d{v}/d{indep} = "))
            def system_f(t_val, y_vec):
                local_env = {**math.__dict__, 'np': np, 't': t_val}
                for i, v in enumerate(vars_list): local_env[v] = y_vec[i]
                res = []
                for e in exprs:
                    try: res.append(eval(e, {"__builtins__": None}, local_env))
                    except: res.append(0.0)
                return np.array(res)
            return system_f, str(exprs)

        else:
            expr = input("y' = ")
            def scalar_f(x, y):
                return eval(expr, {**math.__dict__, 'np': np, 'x': x, 'y': y})
            return scalar_f, expr

    def get_parameters(self, context):
        # ... (Giữ nguyên logic nhập tham số) ...
        print("\n--- BƯỚC 3: THAM SỐ & HIỂN THỊ ---")
        t0 = float(input(f"Giá trị đầu {context['indep']}0: "))

        y0 = []
        if context['type'] == 'scalar':
            y0 = float(input(f"y({t0}) = "))
        elif context['type'] == 'high_order':
            print("Nhập điều kiện đầu:")
            y0.append(float(input(f"  y({t0}) = ")))
            y0.append(float(input(f"  y'({t0}) = ")))
            for i in range(2, context['dim']): y0.append(float(input(f"  y^({i})({t0}) = ")))
            y0 = np.array(y0)
        else:
            print("Nhập giá trị đầu:")
            for v in context['vars']: y0.append(float(input(f"  {v}({t0}) = ")))
            y0 = np.array(y0)

        t_end = float(input(f"Giá trị cuối {context['indep']}_end: "))
        h = float(input("Bước nhảy h: "))

        # MỚI: Hỏi số bước cần hiển thị chi tiết
        detail_steps = int(input("Bạn muốn hiển thị công thức thay số cho bao nhiêu bước đầu? (Nhập 0 để ẩn, 5 để xem 5 bước): ") or "0")

        return t0, y0, t_end, h, detail_steps

    # --- PHẦN QUAN TRỌNG: HÀM IN CÔNG THỨC ---
    def print_formula_explanation(self, method, step, t_old, y_old, t_new, y_new, h, f_val, context):
        """In ra công thức toán học và quá trình thay số"""

        indep = context['indep']
        vars_names = context['vars']
        is_scalar = context['type'] == 'scalar'

        print(f"\n--- Bước {step} (từ {indep}_{step-1}={t_old:.4f} đến {indep}_{step}={t_new:.4f}) ---")

        # Xử lý vector/scalar để in ấn
        if is_scalar:
            y_old_disp = [y_old]
            y_new_disp = [y_new]
            f_val_disp = [f_val] if np.isscalar(f_val) else f_val
        else:
            y_old_disp = y_old
            y_new_disp = y_new
            f_val_disp = f_val

        # Duyệt qua từng biến (ví dụ: x, y hoặc y, y')
        for i, var_name in enumerate(vars_names):
            val_old = y_old_disp[i]
            val_new = y_new_disp[i]
            val_f   = f_val_disp[i]

            # 1. EULER HIỆN: y_new = y_old + h * f(...)
            if method == '1':
                print(f"  Biến {var_name}:")
                print(f"    Công thức: {var_name}_{step} = {var_name}_{step-1} + h * f_{i}(...)")
                print(f"    Thay số:   {var_name}_{step} = {val_old:.6f} + {h} * ({val_f:.6f})")
                print(f"    Kết quả:   {var_name}_{step} = {val_new:.6f}")

            # 2. EULER ẨN: Cần giải pt
            elif method == '2':
                print(f"  Biến {var_name}:")
                print(f"    Công thức: {var_name}_{step} = {var_name}_{step-1} + h * f({indep}_{step}, ...)")
                print(f"    (Phương pháp ẩn cần giải phương trình phi tuyến để tìm {var_name}_{step})")
                print(f"    Kết quả tìm được: {var_name}_{step} = {val_new:.6f}")

            # 3. HÌNH THANG: y_new = y_old + h/2 * (f_old + f_new)
            elif method == '3':
                print(f"  Biến {var_name}:")
                print(f"    Công thức: {var_name}_{step} = {var_name}_{step-1} + (h/2) * [f_{step-1} + f_{step}]")
                print(f"    (Dùng fsolve giải phương trình)")
                print(f"    Kết quả:   {var_name}_{step} = {val_new:.6f}")

    def run_solver(self, method, f, t0, y0, t_end, h, context, detail_steps):
        t_vals = [t0]
        y_vals = [y0]
        t, y = t0, y0
        is_vector = isinstance(y0, np.ndarray)

        # In công thức tổng quát ban đầu
        print("\n" + "*"*60)
        print("CÔNG THỨC TỔNG QUÁT:")
        if method == '1':
            print(f"  y_(n+1) = y_n + h * f(t_n, y_n)")
        elif method == '2':
            print(f"  y_(n+1) = y_n + h * f(t_(n+1), y_(n+1))  (Giải phương trình)")
        elif method == '3':
            print(f"  y_(n+1) = y_n + (h/2) * [f(t_n, y_n) + f(t_(n+1), y_(n+1))]")
        print("*"*60)

        step = 1
        while t < t_end - h/10:
            # Lưu giá trị cũ để in ấn
            t_old = t
            y_old = y if is_vector else float(y)
            f_old_val = f(t, y) # Tính f tại bước cũ (cho Euler hiện)
            if not is_vector and isinstance(f_old_val, np.ndarray): f_old_val = f_old_val[0]

            # --- TÍNH TOÁN (CORE) ---
            if method == '1':
                y_new = y + h * f_old_val

            elif method == '2':
                t_new = t + h
                guess = y + h * f_old_val
                def eq(yi):
                    fi = f(t_new, yi)
                    if not is_vector and isinstance(fi, np.ndarray): fi = fi[0]
                    return yi - y - h * fi
                y_new = fsolve(eq, guess) if is_vector else fsolve(eq, guess)[0]

            elif method == '3':
                t_new = t + h
                guess = y + h * f_old_val
                def eq(yi):
                    fi = f(t_new, yi)
                    if not is_vector and isinstance(fi, np.ndarray): fi = fi[0]
                    return yi - y - (h/2)*(f_old_val + fi)
                y_new = fsolve(eq, guess) if is_vector else fsolve(eq, guess)[0]

            # --- HIỂN THỊ CÔNG THỨC (NẾU CẦN) ---
            if step <= detail_steps:
                # Với Euler hiện, ta dùng f_old_val để in
                # Với phương pháp khác, ta chỉ in kết quả vì quá trình giải fsolve rất phức tạp để in
                self.print_formula_explanation(method, step, t_old, y_old, t + h, y_new, h, f_old_val, context)

            # Cập nhật
            t += h
            t_vals.append(t)
            y_vals.append(y_new if not is_vector else y_new.copy())
            y = y_new
            step += 1

        return np.array(t_vals), np.array(y_vals)

    def run(self):
        self.display_welcome()
        ctx = self.get_problem_context()
        f, expr_str = self.get_function(ctx)

        print("\nCHỌN PHƯƠNG PHÁP:")
        for k,v in self.methods.items(): print(f"{k}. {v}")
        m = input("Chọn: ")

        t0, y0, t_end, h, d_steps = self.get_parameters(ctx)

        print("\n🔄 ĐANG TÍNH TOÁN...")
        ts, ys = self.run_solver(m, f, t0, y0, t_end, h, ctx, d_steps)

        print("\n" + "="*40)
        print(f"KẾT QUẢ TẠI t = {ts[-1]:.4f}")
        if ctx['type'] == 'high_order':
            print(f"y = {ys[-1][0]:.6f}")
            print(f"y' = {ys[-1][1]:.6f}")
        elif ctx['type'] == 'scalar':
            print(f"y = {ys[-1]:.6f}")
        else:
            for i, v in enumerate(ctx['vars']):
                print(f"{v} = {ys[-1][i]:.6f}")
        print("="*40)

        # Vẽ đồ thị (như cũ)
        plt.figure(figsize=(10, 6))
        if ctx['type'] == 'scalar': plt.plot(ts, ys, 'b-o', label='y')
        elif ctx['type'] == 'high_order':
            plt.plot(ts, ys[:,0], 'b-', label='y (nghiệm)')
            plt.plot(ts, ys[:,1], 'r--', label="y' (đạo hàm)", alpha=0.5)
        else:
            for i, v in enumerate(ctx['vars']): plt.plot(ts, ys[:, i], label=v)
        plt.legend(); plt.grid(True); plt.title(f"Đồ thị nghiệm ({self.methods[m]})"); plt.show()

if __name__ == "__main__":
    EulerSolverExplainer().run()