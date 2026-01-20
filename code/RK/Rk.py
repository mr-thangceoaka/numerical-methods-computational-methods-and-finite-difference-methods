#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phân tích lý thuyết phương pháp Runge-Kutta
Sử dụng SymPy để làm việc với biểu thức symbolic
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sympy as sp
from sympy import symbols, expand, simplify, latex, lambdify

# Thiết lập
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)


class RungeKuttaTheory:
    """Lớp phân tích lý thuyết Runge-Kutta"""

    def __init__(self, order, alpha_values=None):
        self.order = order
        self.alpha_values = alpha_values
        self.coefficients = None

        # Symbolic variables
        self.z = symbols('z', complex=True)
        self.h, self.lam = symbols('h lambda', real=True)

        self.build_rk_formula()
        self.compute_stability_function()

    def build_rk_formula(self):
        """Xây dựng công thức RK và in kết quả"""
        print("\n" + "="*80)
        print(f"XÂY DỰNG CÔNG THỨC RUNGE-KUTTA BẬC {self.order}")
        print("="*80)

        if self.order == 1:
            self.build_rk1()
        elif self.order == 2:
            self.build_rk2()
        elif self.order == 3:
            self.build_rk3()
        elif self.order == 4:
            self.build_rk4()

    def build_rk1(self):
        """RK1 - Euler hiện"""
        self.coefficients = {
            'r': [1.0],
            'alpha': [0.0],
            'beta': [[0.0]],
            'stages': 1
        }

        print("\n📌 CÔNG THỨC: y_{n+1} = y_n + h·f(x_n, y_n)")
        print("\n📊 HỆ SỐ:")
        print(f"   r₁ = {self.coefficients['r'][0]}")

    def build_rk2(self):
        """RK2 với tham số α₂"""
        alpha2 = self.alpha_values[0] if self.alpha_values else 0.5

        if alpha2 == 0:
            print("⚠️  α₂ = 0 không hợp lệ! Sử dụng α₂ = 0.5")
            alpha2 = 0.5

        print(f"\n📌 THAM SỐ ĐẦU VÀO: α₂ = {alpha2}")
        print("\n🔢 GIẢI HỆ ĐIỀU KIỆN RK2:")
        print("   (1) r₁ + r₂ = 1")
        print("   (2) r₂·α₂ = 1/2")
        print("   (3) β₁₁ = α₂")

        r2 = 1.0 / (2.0 * alpha2)
        r1 = 1.0 - r2
        beta11 = alpha2

        self.coefficients = {
            'r': [r1, r2],
            'alpha': [0.0, alpha2],
            'beta': [[0.0], [beta11]],
            'stages': 2
        }

        print(f"\n✅ KẾT QUẢ:")
        print(f"   r₁ = {r1:.6f}")
        print(f"   r₂ = {r2:.6f}")
        print(f"   α₂ = {alpha2}")
        print(f"   β₁₁ = {beta11}")

        print(f"\n✓ KIỂM TRA:")
        print(f"   r₁ + r₂ = {r1 + r2:.10f} (= 1 ✓)")
        print(f"   r₂·α₂ = {r2*alpha2:.10f} (= 0.5 ✓)")

        print(f"\n📌 CÔNG THỨC:")
        print(f"   k₁ = h·f(xₙ, yₙ)")
        print(f"   k₂ = h·f(xₙ + {alpha2}h, yₙ + {beta11}k₁)")
        print(f"   y_{{n+1}} = yₙ + {r1}k₁ + {r2}k₂")

    def build_rk3(self):
        """RK3 với tham số α₂, α₃"""
        alpha2 = self.alpha_values[0] if self.alpha_values and len(self.alpha_values) > 0 else 0.5
        alpha3 = self.alpha_values[1] if self.alpha_values and len(self.alpha_values) > 1 else 1.0

        print(f"\n📌 THAM SỐ ĐẦU VÀO: α₂ = {alpha2}, α₃ = {alpha3}")
        print("\n🔢 GIẢI HỆ ĐIỀU KIỆN RK3 (6 phương trình, 6 ẩn):")
        print("   (1) r₁ + r₂ + r₃ = 1")
        print("   (2) r₂·α₂ + r₃·α₃ = 1/2")
        print("   (3) r₂·α₂² + r₃·α₃² = 1/3")
        print("   (4) r₃·β₂₁·α₂ = 1/6")
        print("   (5) α₂ = β₁₁")
        print("   (6) α₃ = β₂₁ + β₂₂")

        def equations(vars):
            r1, r2, r3, beta11, beta21, beta22 = vars
            return [
                r1 + r2 + r3 - 1,
                r2*alpha2 + r3*alpha3 - 0.5,
                r2*alpha2**2 + r3*alpha3**2 - 1/3,
                r3*beta21*alpha2 - 1/6,
                alpha2 - beta11,
                alpha3 - beta21 - beta22
            ]

        initial_guess = [1/6, 2/3, 1/6, alpha2, 0, alpha3]
        solution = fsolve(equations, initial_guess)
        r1, r2, r3, beta11, beta21, beta22 = solution

        self.coefficients = {
            'r': [r1, r2, r3],
            'alpha': [0.0, alpha2, alpha3],
            'beta': [[0.0], [beta11], [beta21, beta22]],
            'stages': 3
        }

        print(f"\n✅ KẾT QUẢ:")
        print(f"   r₁ = {r1:.10f}")
        print(f"   r₂ = {r2:.10f}")
        print(f"   r₃ = {r3:.10f}")
        print(f"   β₁₁ = {beta11:.10f}")
        print(f"   β₂₁ = {beta21:.10f}")
        print(f"   β₂₂ = {beta22:.10f}")

        print(f"\n✓ KIỂM TRA:")
        print(f"   r₁ + r₂ + r₃ = {r1+r2+r3:.10f} (= 1 ✓)")
        print(f"   r₂·α₂ + r₃·α₃ = {r2*alpha2 + r3*alpha3:.10f} (= 0.5 ✓)")
        print(f"   r₂·α₂² + r₃·α₃² = {r2*alpha2**2 + r3*alpha3**2:.10f} (= 0.333... ✓)")

    def build_rk4(self):
        """RK4 cổ điển"""
        self.coefficients = {
            'r': [1/6, 1/3, 1/3, 1/6],
            'alpha': [0.0, 0.5, 0.5, 1.0],
            'beta': [[0.0], [0.5], [0.0, 0.5], [0.0, 0.0, 1.0]],
            'stages': 4
        }

        print("\n📌 CÔNG THỨC RK4 CỔ ĐIỂN:")
        print("   k₁ = h·f(xₙ, yₙ)")
        print("   k₂ = h·f(xₙ + h/2, yₙ + k₁/2)")
        print("   k₃ = h·f(xₙ + h/2, yₙ + k₂/2)")
        print("   k₄ = h·f(xₙ + h, yₙ + k₃)")
        print("   y_{n+1} = yₙ + (k₁ + 2k₂ + 2k₃ + k₄)/6")

    def compute_stability_function(self):
        """Tính hàm ổn định R(z) - cả symbolic và numerical"""
        print("\n" + "="*80)
        print("HÀM ỔN ĐỊNH R(z) - PHÂN TÍCH LÝ THUYẾT")
        print("="*80)

        print("\n📖 Với phương trình test: y' = λy")
        print("   Hệ số khuếch đại: y_{n+1} = R(z)·y_n, với z = h·λ")

        z = self.z

        if self.order == 1:
            # RK1: R(z) = 1 + z
            self.R_symbolic = 1 + z

        elif self.order == 2:
            # RK2: R(z) = 1 + z + r₂·α₂·z²
            r2 = self.coefficients['r'][1]
            alpha2 = self.coefficients['alpha'][1]
            self.R_symbolic = 1 + z + r2*alpha2*z**2

        elif self.order == 3:
            # RK3: R(z) = 1 + z + z²/2 + r₃·β₂₁·α₂·z³
            r3 = self.coefficients['r'][2]
            beta21 = self.coefficients['beta'][2][0]
            alpha2 = self.coefficients['alpha'][1]
            self.R_symbolic = 1 + z + z**2/2 + r3*beta21*alpha2*z**3

        elif self.order == 4:
            # RK4: R(z) = 1 + z + z²/2 + z³/6 + z⁴/24
            self.R_symbolic = 1 + z + z**2/2 + z**3/6 + z**4/24

        # Simplify và expand
        self.R_simplified = simplify(expand(self.R_symbolic))

        print(f"\n🔢 HÀM Ổn ĐỊNH (SYMBOLIC):")
        print(f"   R(z) = {self.R_simplified}")

        # Tạo hàm numerical
        self.R_numerical = lambdify(z, self.R_simplified, 'numpy')

        # Phân tích hệ số
        if self.order <= 4:
            poly_coeffs = [self.R_simplified.as_coefficients_dict()[z**i]
                           if z**i in self.R_simplified.as_coefficients_dict()
                           else 0 for i in range(self.order + 1)]

            print(f"\n📊 KHAI TRIỂN TAYLOR:")
            for i, coef in enumerate(poly_coeffs):
                if i == 0:
                    print(f"   R(z) = {coef}", end="")
                else:
                    print(f" + ({coef})·z^{i}", end="")
            print()

    def analyze_convergence_order(self, f_symbolic, y0_val, x_range=(0, 1)):
        """
        Phân tích cấp hội tụ với hàm symbolic và numerical

        Parameters:
        -----------
        f_symbolic : sympy expression hoặc callable
            Hàm f(x,y) symbolic hoặc lambda
        y0_val : float
            Giá trị đầu
        x_range : tuple
            Khoảng tính toán (x0, x_end)
        """
        print("\n" + "="*80)
        print("PHÂN TÍCH HỘI TỤ VÀ CẤP HỘI TỤ")
        print("="*80)

        x0, x_end = x_range

        # Nếu f là symbolic, convert sang numerical
        if hasattr(f_symbolic, 'free_symbols'):
            x_sym, y_sym = symbols('x y', real=True)
            f_num = lambdify((x_sym, y_sym), f_symbolic, 'numpy')
            print(f"\n📌 Hàm f(x,y) = {f_symbolic}")
        else:
            f_num = f_symbolic
            print(f"\n📌 Hàm f(x,y): numerical function")

        print(f"   Điều kiện đầu: y({x0}) = {y0_val}")
        print(f"   Khoảng tính: [{x0}, {x_end}]")

        # Các bước khác nhau
        h_values = [0.1, 0.05, 0.025, 0.0125]
        errors = []

        print(f"\n{'h':<12} {'y(x_end)':<18} {'Sai số ước lượng':<20} {'Tỷ lệ':<12}")
        print("-" * 70)

        y_prev = None
        for i, h in enumerate(h_values):
            n_steps = int((x_end - x0) / h) + 1
            x_vals = np.linspace(x0, x_end, n_steps)
            y_vals = np.zeros(n_steps)
            y_vals[0] = y0_val

            for j in range(n_steps - 1):
                y_vals[j+1] = self.apply_step(f_num, x_vals[j], y_vals[j], h)

            y_end = y_vals[-1]

            if y_prev is not None:
                # Ước lượng sai số bằng Richardson extrapolation
                error_est = abs(y_end - y_prev) / (2**self.order - 1)
                errors.append(error_est)

                if len(errors) > 1:
                    ratio = errors[-2] / errors[-1]
                    print(f"{h:<12.5f} {y_end:<18.10f} {error_est:<20.6e} {ratio:<12.6f}")
                else:
                    print(f"{h:<12.5f} {y_end:<18.10f} {error_est:<20.6e} {'---':<12}")
            else:
                print(f"{h:<12.5f} {y_end:<18.10f} {'---':<20} {'---':<12}")

            y_prev = y_end

        if len(errors) >= 2:
            # Ước lượng cấp hội tụ
            log_errors = np.log(errors)
            log_h = np.log(h_values[1:len(errors)+1])
            p_est = -np.polyfit(log_h, log_errors, 1)[0]

            print(f"\n📊 CẤP HỘI TỤ:")
            print(f"   Lý thuyết: p = {self.order}")
            print(f"   Ước lượng: p ≈ {p_est:.4f}")

            # Vẽ đồ thị
            plt.figure(figsize=(10, 6))
            plt.loglog(h_values[1:], errors, 'bo-', label='Sai số ước lượng',
                       markersize=10, linewidth=2)

            # Đường tham chiếu
            h_ref = np.array(h_values[1:])
            err_ref = errors[0] * (h_ref / h_values[1])**self.order
            plt.loglog(h_ref, err_ref, 'r--', label=f'Độ dốc = {self.order}', linewidth=2)

            plt.xlabel('Bước nhảy h', fontsize=13)
            plt.ylabel('Sai số ước lượng', fontsize=13)
            plt.title(f'Đồ thị hội tụ - RK{self.order}', fontsize=14, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3, which='both')
            plt.tight_layout()
            plt.savefig(f'/mnt/user-data/outputs/theory_convergence_rk{self.order}.png',
                        dpi=300, bbox_inches='tight')
            print(f"\n✅ Đã lưu: theory_convergence_rk{self.order}.png")
            plt.close()

    def apply_step(self, f, x, y, h):
        """Áp dụng một bước RK"""
        k = []
        for i in range(self.coefficients['stages']):
            x_eval = x + self.coefficients['alpha'][i] * h
            y_eval = y
            if i > 0:
                for j in range(i):
                    if j < len(self.coefficients['beta'][i]):
                        y_eval += self.coefficients['beta'][i][j] * k[j]
            k.append(h * f(x_eval, y_eval))

        y_new = y
        for i in range(self.coefficients['stages']):
            y_new += self.coefficients['r'][i] * k[i]
        return y_new

    def find_stability_boundary(self):
        """Tìm biên miền ổn định"""
        print("\n" + "="*80)
        print("MIỀN ỔN ĐỊNH TUYỆT ĐỐI")
        print("="*80)

        print("\n📖 Định nghĩa: Miền ổn định = {z ∈ ℂ : |R(z)| ≤ 1}")

        # Tìm biên trên trục thực
        real_axis = np.linspace(-10, 2, 2000)
        R_real = np.abs(self.R_numerical(real_axis))
        stable_real = real_axis[R_real <= 1.0]

        if len(stable_real) > 0:
            left_bound = stable_real.min()
            right_bound = stable_real.max()
            print(f"\n📍 Biên trên trục thực:")
            print(f"   Trái: z ≈ {left_bound:.6f}")
            print(f"   Phải: z ≈ {right_bound:.6f}")

        # Tìm biên trên trục ảo
        imag_vals = np.linspace(0, 10, 2000)
        imag_axis = 1j * imag_vals
        R_imag = np.abs(self.R_numerical(imag_axis))
        stable_imag = imag_vals[R_imag <= 1.0]

        if len(stable_imag) > 0:
            top_bound = stable_imag.max()
            print(f"   Trên trục ảo: z ≈ ±{top_bound:.6f}i")

        # Vẽ miền ổn định
        self.plot_stability_region()

    def plot_stability_region(self):
        """Vẽ miền ổn định"""
        x = np.linspace(-5, 2, 800)
        y = np.linspace(-5, 5, 800)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j*Y

        R_vals = np.abs(self.R_numerical(Z))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Đồ thị 1: Miền ổn định
        ax1.contourf(X, Y, R_vals, levels=[0, 1], colors=['lightgreen'], alpha=0.7)
        ax1.contour(X, Y, R_vals, levels=[1], colors=['darkgreen'], linewidths=2.5)
        ax1.axhline(y=0, color='k', linewidth=0.8, linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='k', linewidth=0.8, linestyle='-', alpha=0.3)
        ax1.set_xlabel('Re(z)', fontsize=13)
        ax1.set_ylabel('Im(z)', fontsize=13)
        ax1.set_title(f'Miền ổn định - RK{self.order}\n|R(z)| ≤ 1 (vùng xanh)',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Đồ thị 2: Đường mức
        levels = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        cs = ax2.contour(X, Y, R_vals, levels=levels, linewidths=2)
        ax2.clabel(cs, inline=True, fontsize=11, fmt='%.1f')
        ax2.axhline(y=0, color='k', linewidth=0.8, linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linewidth=0.8, linestyle='-', alpha=0.3)
        ax2.set_xlabel('Re(z)', fontsize=13)
        ax2.set_ylabel('Im(z)', fontsize=13)
        ax2.set_title(f'Đường mức |R(z)| - RK{self.order}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f'/mnt/user-data/outputs/theory_stability_rk{self.order}.png',
                    dpi=300, bbox_inches='tight')
        print(f"\n✅ Đã lưu: theory_stability_rk{self.order}.png")
        plt.close()


def main():
    """Chương trình chính"""
    print("\n" + "="*80)
    print(" "*20 + "PHÂN TÍCH LÝ THUYẾT RUNGE-KUTTA")
    print("="*80)

    # Nhập thông tin
    while True:
        try:
            order = int(input("\nBậc RK (1/2/3/4): "))
            if order in [1, 2, 3, 4]:
                break
            print("⚠️  Chọn 1, 2, 3 hoặc 4!")
        except:
            print("⚠️  Nhập số nguyên!")

    # Nhập alpha nếu cần
    alpha_values = None
    if order == 2:
        alpha_str = input("α₂ (Enter = 0.5): ").strip()
        alpha2 = float(alpha_str) if alpha_str else 0.5
        alpha_values = [alpha2]
    elif order == 3:
        alpha2_str = input("α₂ (Enter = 0.5): ").strip()
        alpha3_str = input("α₃ (Enter = 1.0): ").strip()
        alpha2 = float(alpha2_str) if alpha2_str else 0.5
        alpha3 = float(alpha3_str) if alpha3_str else 1.0
        alpha_values = [alpha2, alpha3]

    # Tạo analyzer
    rk = RungeKuttaTheory(order, alpha_values)

    # Phân tích miền ổn định
    rk.find_stability_boundary()

    # Chọn hàm để test
    print("\n" + "="*80)
    print("CHỌN HÀM ĐỂ KHẢO SÁT HỘI TỤ")
    print("="*80)
    print("1. y' = -y (lý thuyết: y = e^(-x))")
    print("2. y' = y (lý thuyết: y = e^x)")
    print("3. y' = x (lý thuyết: y = x²/2)")
    print("4. y' = -2xy (lý thuyết: y = e^(-x²))")

    while True:
        try:
            choice = int(input("\nChọn (1-4): "))
            if choice in [1, 2, 3, 4]:
                break
        except:
            pass

    # Tạo hàm
    x, y = symbols('x y', real=True)
    if choice == 1:
        f = -y
        y0 = 1.0
        x_range = (0, 2)
    elif choice == 2:
        f = y
        y0 = 1.0
        x_range = (0, 2)
    elif choice == 3:
        f = x
        y0 = 0.0
        x_range = (0, 2)
    else:
        f = -2*x*y
        y0 = 1.0
        x_range = (0, 2)

    # Phân tích hội tụ
    rk.analyze_convergence_order(f, y0, x_range)

    print("\n" + "="*80)
    print("HOÀN THÀNH!")
    print("="*80)
    print(f"\nĐã tạo các file:")
    print(f"  - theory_convergence_rk{order}.png")
    print(f"  - theory_stability_rk{order}.png")


if __name__ == "__main__":
    main()