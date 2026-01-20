#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
RUNGE-KUTTA SOLVER - Giải bài toán Cauchy cho phương trình vi phân thường
==================================================================================

Tác giả: Numerical Analysis Expert
Mô tả: Cài đặt tổng quát các phương pháp Runge-Kutta hiện (ERK) dựa trên
       Bảng Butcher (Butcher Tableau)

Công thức tổng quát:
    k_i = h·f(x_n + α_i·h, y_n + Σ(β_ij·k_j))  với j=1..i-1
    y_{n+1} = y_n + Σ(r_i·k_i)                 với i=1..s

Trong đó:
    - s: Số nấc (stages)
    - p: Bậc chính xác (order of accuracy)
    - α: Vector hệ số cho biến x (size s)
    - β: Ma trận hệ số cho các k_j (size s×s, tam giác dưới)
    - r: Vector trọng số cho y_{n+1} (size s)

Tham khảo:
    - Butcher, J.C. (2008). Numerical Methods for Ordinary Differential Equations
    - Hairer, E., Nørsett, S.P., Wanner, G. (1993). Solving ODEs I
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict, List
from dataclasses import dataclass
import warnings

# Thiết lập matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


@dataclass
class ButcherTableau:
    """
    Cấu trúc dữ liệu cho Bảng Butcher

    Attributes:
    -----------
    name : str
        Tên phương pháp
    s : int
        Số nấc (stages)
    p : int
        Bậc chính xác (order)
    alpha : np.ndarray
        Vector α (size s) - hệ số cho x_n
    beta : np.ndarray
        Ma trận β (size s×s) - hệ số cho k_j (tam giác dưới)
    r : np.ndarray
        Vector r (size s) - trọng số cho y_{n+1}
    """
    name: str
    s: int
    p: int
    alpha: np.ndarray
    beta: np.ndarray
    r: np.ndarray

    def __post_init__(self):
        """Kiểm tra tính hợp lệ của Bảng Butcher"""
        # Kiểm tra kích thước
        assert len(self.alpha) == self.s, f"alpha phải có kích thước {self.s}"
        assert self.beta.shape == (self.s, self.s), f"beta phải có kích thước {self.s}×{self.s}"
        assert len(self.r) == self.s, f"r phải có kích thước {self.s}"

        # Kiểm tra điều kiện consistency: α_i = Σβ_ij
        for i in range(self.s):
            sum_beta = np.sum(self.beta[i, :i])
            if not np.isclose(self.alpha[i], sum_beta):
                warnings.warn(f"Hàng {i}: α[{i}]={self.alpha[i]} ≠ Σβ[{i},j]={sum_beta}")

        # Kiểm tra tổng trọng số
        sum_r = np.sum(self.r)
        if not np.isclose(sum_r, 1.0):
            warnings.warn(f"Tổng trọng số Σr_i = {sum_r} ≠ 1")

    def __repr__(self):
        return f"ButcherTableau(name='{self.name}', s={self.s}, p={self.p})"


class ButcherLibrary:
    """
    Thư viện các phương pháp Runge-Kutta tiêu chuẩn
    Định nghĩa các Bảng Butcher cho các phương pháp ERK phổ biến
    """

    @staticmethod
    def get_method(method_name: str) -> ButcherTableau:
        """
        Lấy Bảng Butcher theo tên phương pháp

        Parameters:
        -----------
        method_name : str
            Tên phương pháp (không phân biệt hoa/thường)

        Returns:
        --------
        ButcherTableau
        """
        methods = {
            # ===== RK1 =====
            'RK1': ButcherLibrary._rk1_euler,
            'EULER': ButcherLibrary._rk1_euler,

            # ===== RK2 =====
            'RK2_HEUN': ButcherLibrary._rk2_heun,
            'RK2_MIDPOINT': ButcherLibrary._rk2_midpoint,
            'RK2_RALSTON': ButcherLibrary._rk2_ralston,

            # ===== RK3 =====
            'RK3': ButcherLibrary._rk3_classic,
            'RK3_CLASSIC': ButcherLibrary._rk3_classic,
            'RK3_NYSTROM': ButcherLibrary._rk3_classic,
            'RK3_HEUN': ButcherLibrary._rk3_heun,

            # ===== RK4 =====
            'RK4': ButcherLibrary._rk4_classic,
            'RK4_CLASSIC': ButcherLibrary._rk4_classic,
            'RK4_38': ButcherLibrary._rk4_38rule,

            # ===== RK5 =====
            'RK5': ButcherLibrary._rk5_butcher,
            'RK5_BUTCHER': ButcherLibrary._rk5_butcher,
        }

        key = method_name.upper()
        if key not in methods:
            available = ', '.join(methods.keys())
            raise ValueError(f"Phương pháp '{method_name}' không tồn tại. "
                             f"Các phương pháp có sẵn: {available}")

        return methods[key]()

    @staticmethod
    def _rk1_euler() -> ButcherTableau:
        """
        RK1 - Euler hiện (Forward Euler)
        s=1, p=1
        """
        return ButcherTableau(
            name="RK1 (Euler)",
            s=1, p=1,
            alpha=np.array([0.0]),
            beta=np.array([[0.0]]),
            r=np.array([1.0])
        )

    @staticmethod
    def _rk2_heun() -> ButcherTableau:
        """
        RK2 - Heun (Trapezoidal)
        s=2, p=2
        α = [0, 1], r = [1/2, 1/2]

        Bảng Butcher:
        0  |
        1  | 1
        ---|------
           | 1/2  1/2
        """
        return ButcherTableau(
            name="RK2 (Heun)",
            s=2, p=2,
            alpha=np.array([0.0, 1.0]),
            beta=np.array([
                [0.0, 0.0],
                [1.0, 0.0]
            ]),
            r=np.array([0.5, 0.5])
        )

    @staticmethod
    def _rk2_midpoint() -> ButcherTableau:
        """
        RK2 - Midpoint (Điểm giữa)
        s=2, p=2
        α = [0, 1/2], r = [0, 1]

        Bảng Butcher:
        0   |
        1/2 | 1/2
        ----|------
            | 0    1
        """
        return ButcherTableau(
            name="RK2 (Midpoint)",
            s=2, p=2,
            alpha=np.array([0.0, 0.5]),
            beta=np.array([
                [0.0, 0.0],
                [0.5, 0.0]
            ]),
            r=np.array([0.0, 1.0])
        )

    @staticmethod
    def _rk2_ralston() -> ButcherTableau:
        """
        RK2 - Ralston (Tối ưu hóa sai số cắt)
        s=2, p=2
        α = [0, 2/3], r = [1/4, 3/4], β_21 = 2/3

        Bảng Butcher:
        0   |
        2/3 | 2/3
        ----|------
            | 1/4  3/4

        Nguồn: Ralston, A. (1962). "Runge-Kutta Methods with Minimum Error Bounds"
        """
        return ButcherTableau(
            name="RK2 (Ralston)",
            s=2, p=2,
            alpha=np.array([0.0, 2.0/3.0]),
            beta=np.array([
                [0.0,     0.0],
                [2.0/3.0, 0.0]
            ]),
            r=np.array([0.25, 0.75])
        )

    @staticmethod
    def _rk3_classic() -> ButcherTableau:
        """
        RK3 - Classic/Nystrom (Thường dùng)
        s=3, p=3
        α = [0, 1/2, 1], r = [1/6, 2/3, 1/6]

        Bảng Butcher:
        0   |
        1/2 | 1/2
        1   | -1    2
        ----|-------------
            | 1/6  2/3  1/6
        """
        return ButcherTableau(
            name="RK3 (Classic/Nystrom)",
            s=3, p=3,
            alpha=np.array([0.0, 0.5, 1.0]),
            beta=np.array([
                [0.0,  0.0, 0.0],
                [0.5,  0.0, 0.0],
                [-1.0, 2.0, 0.0]
            ]),
            r=np.array([1.0/6.0, 2.0/3.0, 1.0/6.0])
        )

    @staticmethod
    def _rk3_heun() -> ButcherTableau:
        """
        RK3 - Heun (Lưu ý: r_2 = 0)
        s=3, p=3
        α = [0, 1/3, 2/3], r = [1/4, 0, 3/4]

        Bảng Butcher:
        0   |
        1/3 | 1/3
        2/3 | 0    2/3
        ----|-------------
            | 1/4  0    3/4
        """
        return ButcherTableau(
            name="RK3 (Heun)",
            s=3, p=3,
            alpha=np.array([0.0, 1.0/3.0, 2.0/3.0]),
            beta=np.array([
                [0.0,     0.0,     0.0],
                [1.0/3.0, 0.0,     0.0],
                [0.0,     2.0/3.0, 0.0]
            ]),
            r=np.array([0.25, 0.0, 0.75])
        )

    @staticmethod
    def _rk4_classic() -> ButcherTableau:
        """
        RK4 - Classic (Thường dùng)
        s=4, p=4
        Quy tắc 1/6: r = [1/6, 2/6, 2/6, 1/6]

        Bảng Butcher:
        0   |
        1/2 | 1/2
        1/2 | 0    1/2
        1   | 0    0    1
        ----|----------------
            | 1/6  1/3  1/3  1/6
        """
        return ButcherTableau(
            name="RK4 (Classic)",
            s=4, p=4,
            alpha=np.array([0.0, 0.5, 0.5, 1.0]),
            beta=np.array([
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ]),
            r=np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
        )

    @staticmethod
    def _rk4_38rule() -> ButcherTableau:
        """
        RK4 - 3/8 Rule (Quy tắc 3/8)
        s=4, p=4
        α = [0, 1/3, 2/3, 1], r = [1/8, 3/8, 3/8, 1/8]

        Bảng Butcher:
        0   |
        1/3 | 1/3
        2/3 | -1/3  1
        1   | 1     -1    1
        ----|--------------------
            | 1/8   3/8   3/8  1/8
        """
        return ButcherTableau(
            name="RK4 (3/8 Rule)",
            s=4, p=4,
            alpha=np.array([0.0, 1.0/3.0, 2.0/3.0, 1.0]),
            beta=np.array([
                [0.0,      0.0,  0.0, 0.0],
                [1.0/3.0,  0.0,  0.0, 0.0],
                [-1.0/3.0, 1.0,  0.0, 0.0],
                [1.0,     -1.0,  1.0, 0.0]
            ]),
            r=np.array([1.0/8.0, 3.0/8.0, 3.0/8.0, 1.0/8.0])
        )

    @staticmethod
    def _rk5_butcher() -> ButcherTableau:
        """
        RK5 - Butcher's method
        s=6, p=5 (Lưu ý: cần 6 nấc để đạt bậc 5)

        Bảng Butcher theo Butcher (1964):
        0    |
        1/4  | 1/4
        1/4  | 1/8   1/8
        1/2  | 0    -1/2   1
        3/4  | 3/16  0     0     9/16
        1    | -3/7  8/7   6/7  -12/7  8/7
        -----|----------------------------------------
             | 7/90  0     16/45 2/15  16/45  7/90
        """
        return ButcherTableau(
            name="RK5 (Butcher)",
            s=6, p=5,
            alpha=np.array([0.0, 0.25, 0.25, 0.5, 0.75, 1.0]),
            beta=np.array([
                [0.0,      0.0,     0.0,     0.0,      0.0,     0.0],
                [0.25,     0.0,     0.0,     0.0,      0.0,     0.0],
                [0.125,    0.125,   0.0,     0.0,      0.0,     0.0],
                [0.0,     -0.5,     1.0,     0.0,      0.0,     0.0],
                [3.0/16.0, 0.0,     0.0,     9.0/16.0, 0.0,     0.0],
                [-3.0/7.0, 8.0/7.0, 6.0/7.0,-12.0/7.0, 8.0/7.0, 0.0]
            ]),
            r=np.array([7.0/90.0, 0.0, 16.0/45.0, 2.0/15.0, 16.0/45.0, 7.0/90.0])
        )


class RungeKuttaSolver:
    """
    Solver tổng quát cho bài toán Cauchy sử dụng phương pháp Runge-Kutta hiện

    Bài toán: y' = f(x, y), y(x0) = y0
    """

    def __init__(self, tableau: ButcherTableau):
        """
        Khởi tạo solver với Bảng Butcher

        Parameters:
        -----------
        tableau : ButcherTableau
            Bảng Butcher của phương pháp RK
        """
        self.tableau = tableau
        self.history = None

    def solve(self,
              f: Callable[[float, float], float],
              x0: float,
              y0: float,
              x_end: float,
              h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Giải bài toán Cauchy

        Parameters:
        -----------
        f : callable
            Hàm f(x, y) trong phương trình y' = f(x, y)
        x0 : float
            Điểm bắt đầu
        y0 : float
            Giá trị đầu y(x0)
        x_end : float
            Điểm kết thúc
        h : float
            Bước nhảy

        Returns:
        --------
        x_vals : np.ndarray
            Mảng các giá trị x
        y_vals : np.ndarray
            Mảng các giá trị y tương ứng
        """
        # Kiểm tra điều kiện
        if x_end <= x0:
            raise ValueError(f"x_end ({x_end}) phải lớn hơn x0 ({x0})!")

        # Tính số bước
        n_steps = int(np.ceil((x_end - x0) / h)) + 1
        x_vals = np.linspace(x0, x_end, n_steps)
        y_vals = np.zeros(n_steps)
        y_vals[0] = y0

        # Điều chỉnh h cho chính xác
        h_actual = (x_end - x0) / (n_steps - 1)

        # Giải từng bước
        for i in range(n_steps - 1):
            y_vals[i + 1] = self._step(f, x_vals[i], y_vals[i], h_actual)

        # Lưu lịch sử
        self.history = {'x': x_vals, 'y': y_vals, 'h': h_actual}

        return x_vals, y_vals

    def _step(self, f: Callable, x: float, y: float, h: float) -> float:
        """
        Thực hiện một bước RK theo công thức tổng quát

        Công thức:
            k_i = h·f(x + α_i·h, y + Σ(β_ij·k_j))  với j=1..i-1
            y_new = y + Σ(r_i·k_i)                 với i=1..s

        Parameters:
        -----------
        f : callable
            Hàm f(x, y)
        x, y : float
            Giá trị hiện tại
        h : float
            Bước nhảy

        Returns:
        --------
        y_new : float
            Giá trị mới y_{n+1}
        """
        # Tính các k_i
        k = np.zeros(self.tableau.s)

        for i in range(self.tableau.s):
            # Tính x_i = x + α_i·h
            x_i = x + self.tableau.alpha[i] * h

            # Tính y_i = y + Σ(β_ij·k_j) với j < i
            y_i = y
            for j in range(i):
                y_i += self.tableau.beta[i, j] * k[j]

            # Tính k_i = h·f(x_i, y_i)
            k[i] = h * f(x_i, y_i)

        # Tính y_new = y + Σ(r_i·k_i)
        y_new = y + np.dot(self.tableau.r, k)

        return y_new

    def stability_function(self, z: complex) -> complex:
        """
        Tính hàm ổn định R(z) cho phương trình test y' = λy

        Parameters:
        -----------
        z : complex
            z = h·λ

        Returns:
        --------
        R : complex
            Hệ số khuếch đại
        """
        s = self.tableau.s

        # Xây dựng hệ phương trình tuyến tính: (I - z·β)·k = z·1
        # với 1 là vector [1, 1, ..., 1]
        I = np.eye(s)
        A = I - z * self.tableau.beta
        b = z * np.ones(s)

        # Giải hệ: k = (I - z·β)^{-1}·z·1
        try:
            k = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.inf

        # R(z) = 1 + Σ(r_i·k_i)
        R = 1.0 + np.dot(self.tableau.r, k)

        return R


class RKAnalyzer:
    """
    Module phân tích cho phương pháp Runge-Kutta
    - Kiểm tra hội tụ
    - Vẽ miền ổn định
    """

    @staticmethod
    def convergence_test(solver: RungeKuttaSolver,
                         f: Callable,
                         y_exact: Callable,
                         x0: float,
                         y0: float,
                         x_end: float,
                         h_base: float = 0.1,
                         n_refinements: int = 4) -> Dict:
        """
        Kiểm tra sự hội tụ bằng cách giảm dần bước nhảy

        Parameters:
        -----------
        solver : RungeKuttaSolver
        f : callable
            Hàm f(x, y)
        y_exact : callable
            Nghiệm chính xác y(x)
        x0, y0 : float
            Điều kiện đầu
        x_end : float
            Điểm kết thúc
        h_base : float
            Bước nhảy cơ sở
        n_refinements : int
            Số lần làm mịn (h, h/2, h/4, ...)

        Returns:
        --------
        results : dict
            Chứa h_values, errors, eoc (Empirical Order of Convergence)
        """
        h_values = []
        errors = []

        print("\n" + "="*80)
        print(f"KIỂM TRA HỘI TỤ - {solver.tableau.name}")
        print("="*80)
        print(f"Bậc lý thuyết: p = {solver.tableau.p}")
        print(f"\n{'h':<12} {'Sai số':<15} {'Tỷ lệ':<12} {'EOC':<12}")
        print("-" * 60)

        for i in range(n_refinements):
            h = h_base / (2**i)
            h_values.append(h)

            # Giải
            x_vals, y_vals = solver.solve(f, x0, y0, x_end, h)

            # Tính sai số tại điểm cuối
            error = abs(y_vals[-1] - y_exact(x_end))
            errors.append(error)

            # Tính tỷ lệ và EOC
            if i > 0:
                ratio = errors[i-1] / errors[i]
                eoc = np.log2(ratio)
                print(f"{h:<12.6f} {error:<15.6e} {ratio:<12.6f} {eoc:<12.6f}")
            else:
                print(f"{h:<12.6f} {error:<15.6e} {'---':<12} {'---':<12}")

        # Tính EOC trung bình
        if len(errors) >= 2:
            eoc_values = [np.log2(errors[i]/errors[i+1])
                          for i in range(len(errors)-1)]
            eoc_mean = np.mean(eoc_values)
            print(f"\nEOC trung bình: {eoc_mean:.4f}")
            print(f"So với lý thuyết p={solver.tableau.p}: " +
                  ("✓ Khớp" if abs(eoc_mean - solver.tableau.p) < 0.1
                   else "⚠ Lệch"))
        else:
            eoc_values = []
            eoc_mean = None

        return {
            'h_values': np.array(h_values),
            'errors': np.array(errors),
            'eoc_values': np.array(eoc_values),
            'eoc_mean': eoc_mean
        }

    @staticmethod
    def plot_convergence(results_dict: Dict[str, Dict]):
        """
        Vẽ đồ thị hội tụ cho nhiều phương pháp

        Parameters:
        -----------
        results_dict : dict
            Dictionary {method_name: convergence_results}
        """
        plt.figure(figsize=(12, 6))

        for method_name, results in results_dict.items():
            h_vals = results['h_values']
            errors = results['errors']

            plt.loglog(h_vals, errors, 'o-', label=method_name,
                       markersize=8, linewidth=2)

        plt.xlabel('Bước nhảy h', fontsize=13)
        plt.ylabel('Sai số tại điểm cuối', fontsize=13)
        plt.title('Đồ thị hội tụ các phương pháp RK', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()

        return plt.gcf()

    @staticmethod
    def plot_stability_region(solver: RungeKuttaSolver,
                              xlim: Tuple[float, float] = (-5, 2),
                              ylim: Tuple[float, float] = (-4, 4),
                              resolution: int = 500):
        """
        Vẽ miền ổn định tuyệt đối trên mặt phẳng phức

        Parameters:
        -----------
        solver : RungeKuttaSolver
        xlim, ylim : tuple
            Giới hạn trục Re(z) và Im(z)
        resolution : int
            Độ phân giải lưới
        """
        # Tạo lưới
        re = np.linspace(xlim[0], xlim[1], resolution)
        im = np.linspace(ylim[0], ylim[1], resolution)
        Re, Im = np.meshgrid(re, im)
        Z = Re + 1j*Im

        # Tính |R(z)| cho toàn bộ lưới
        R_abs = np.zeros_like(Z, dtype=float)
        for i in range(resolution):
            for j in range(resolution):
                R_abs[i, j] = abs(solver.stability_function(Z[i, j]))

        # Vẽ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Đồ thị 1: Miền ổn định
        ax1.contourf(Re, Im, R_abs, levels=[0, 1], colors=['lightgreen'], alpha=0.7)
        ax1.contour(Re, Im, R_abs, levels=[1], colors=['darkgreen'], linewidths=2.5)
        ax1.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
        ax1.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
        ax1.set_xlabel('Re(z)', fontsize=13)
        ax1.set_ylabel('Im(z)', fontsize=13)
        ax1.set_title(f'Miền ổn định - {solver.tableau.name}\n|R(z)| ≤ 1 (vùng xanh)',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Đồ thị 2: Đường mức |R(z)|
        levels = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        cs = ax2.contour(Re, Im, R_abs, levels=levels, linewidths=2)
        ax2.clabel(cs, inline=True, fontsize=11)
        ax2.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
        ax2.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
        ax2.set_xlabel('Re(z)', fontsize=13)
        ax2.set_ylabel('Im(z)', fontsize=13)
        ax2.set_title(f'Đường mức |R(z)| - {solver.tableau.name}',
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        plt.tight_layout()

        return fig


class CustomButcherBuilder:
    """Lớp hỗ trợ xây dựng Bảng Butcher tùy chỉnh từ người dùng"""

    @staticmethod
    def build_rk2_custom(alpha2: float) -> ButcherTableau:
        """
        Xây dựng RK2 tùy chỉnh với α₂ do người dùng nhập

        Điều kiện RK2:
        - r₁ + r₂ = 1
        - r₂·α₂ = 1/2
        - β₁₁ = α₂
        """
        if alpha2 == 0:
            raise ValueError("α₂ không được bằng 0!")

        r2 = 1.0 / (2.0 * alpha2)
        r1 = 1.0 - r2

        return ButcherTableau(
            name=f"RK2 (Custom α₂={alpha2})",
            s=2, p=2,
            alpha=np.array([0.0, alpha2]),
            beta=np.array([
                [0.0, 0.0],
                [alpha2, 0.0]
            ]),
            r=np.array([r1, r2])
        )

    @staticmethod
    def build_rk3_custom(alpha2: float, alpha3: float) -> ButcherTableau:
        """Xây dựng RK3 tùy chỉnh (giải hệ phương trình)"""
        from scipy.optimize import fsolve

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

        initial = [1/6, 2/3, 1/6, alpha2, 0, alpha3]
        solution = fsolve(equations, initial)
        r1, r2, r3, beta11, beta21, beta22 = solution

        return ButcherTableau(
            name=f"RK3 (Custom α₂={alpha2}, α₃={alpha3})",
            s=3, p=3,
            alpha=np.array([0.0, alpha2, alpha3]),
            beta=np.array([
                [0.0, 0.0, 0.0],
                [beta11, 0.0, 0.0],
                [beta21, beta22, 0.0]
            ]),
            r=np.array([r1, r2, r3])
        )


def get_user_function():
    """Cho phép người dùng nhập phương trình từ bàn phím"""
    print("\n" + "="*80)
    print("NHẬP PHƯƠNG TRÌNH VI PHÂN")
    print("="*80)

    print("\n📝 Hướng dẫn:")
    print("   - Phương trình dạng: y' = f(x, y)")
    print("   - Sử dụng: x, y, np.exp(), np.sin(), np.cos(), np.sqrt(), etc.")
    print("   - Ví dụ: -y, x*y, x**2 + y, np.sin(x)*y, -2*x*y")

    print("\n🎯 Chọn phương trình:")
    print("   1. y' = -y")
    print("   2. y' = y")
    print("   3. y' = x")
    print("   4. y' = -2*x*y")
    print("   5. Nhập phương trình tùy chỉnh")

    while True:
        try:
            choice = int(input("\nChọn (1-5): "))
            if choice in [1, 2, 3, 4, 5]:
                break
            print("⚠️  Chọn từ 1 đến 5!")
        except:
            print("⚠️  Nhập số nguyên!")

    if choice == 1:
        f_expr = "-y"
        f = lambda x, y: -y
        y_exact = lambda x: np.exp(-x)
        has_exact = True
    elif choice == 2:
        f_expr = "y"
        f = lambda x, y: y
        y_exact = lambda x: np.exp(x)
        has_exact = True
    elif choice == 3:
        f_expr = "x"
        f = lambda x, y: x
        y_exact = lambda x: x**2 / 2
        has_exact = True
    elif choice == 4:
        f_expr = "-2*x*y"
        f = lambda x, y: -2*x*y
        y_exact = lambda x: np.exp(-x**2)
        has_exact = True
    else:
        f_expr = input("\nNhập biểu thức f(x, y): ").strip()
        print(f"\n⚠️  Lưu ý: Bạn đã nhập: f(x, y) = {f_expr}")

        try:
            # Tạo hàm từ biểu thức
            f = eval(f"lambda x, y: {f_expr}")
            # Test thử
            test_val = f(1.0, 1.0)
            print(f"✓ Test: f(1, 1) = {test_val}")

            # Hỏi nghiệm chính xác
            has_exact_input = input("\nBạn có biết nghiệm chính xác không? (y/n): ").lower()
            if has_exact_input == 'y':
                y_expr = input("Nhập y(x) (VD: np.exp(-x)): ").strip()
                y_exact = eval(f"lambda x: {y_expr}")
                test_exact = y_exact(1.0)
                print(f"✓ Test: y(1) = {test_exact}")
                has_exact = True
            else:
                y_exact = None
                has_exact = False

        except Exception as e:
            print(f"❌ Lỗi: {e}")
            print("Sử dụng f(x,y) = -y làm mặc định")
            f_expr = "-y"
            f = lambda x, y: -y
            y_exact = lambda x: np.exp(-x)
            has_exact = True

    return f, f_expr, y_exact, has_exact


def get_initial_conditions():
    """Nhập điều kiện đầu và khoảng tính toán"""
    print("\n" + "="*80)
    print("ĐIỀU KIỆN ĐẦU VÀ KHOẢNG TÍNH TOÁN")
    print("="*80)

    x0 = float(input("\nNhập x₀ (mặc định 0): ") or "0")
    y0 = float(input("Nhập y₀ (mặc định 1): ") or "1")
    x_end = float(input("Nhập x_end (mặc định 2): ") or "2")
    h = float(input("Nhập bước nhảy h (mặc định 0.1): ") or "0.1")

    print(f"\n✓ Điều kiện: y({x0}) = {y0}")
    print(f"✓ Khoảng: [{x0}, {x_end}]")
    print(f"✓ Bước nhảy: h = {h}")

    return x0, y0, x_end, h


def interactive_mode():
    """Chế độ tương tác với người dùng"""
    print("\n" + "="*80)
    print(" "*20 + "RUNGE-KUTTA SOLVER - CHẾ ĐỘ TƯƠNG TÁC")
    print("="*80)

    # Bước 1: Chọn phương pháp
    print("\n" + "="*80)
    print("BƯỚC 1: CHỌN PHƯƠNG PHÁP RUNGE-KUTTA")
    print("="*80)

    print("\n📚 CÁC PHƯƠNG PHÁP CÓ SẴN:")
    print("\n   [RK1]")
    print("   1. RK1 (Euler)")

    print("\n   [RK2]")
    print("   2. RK2_HEUN (α=1)")
    print("   3. RK2_MIDPOINT (α=1/2)")
    print("   4. RK2_RALSTON (α=2/3)")
    print("   5. RK2_CUSTOM (Tự chọn α₂)")

    print("\n   [RK3]")
    print("   6. RK3_CLASSIC (Nystrom)")
    print("   7. RK3_HEUN")
    print("   8. RK3_CUSTOM (Tự chọn α₂, α₃)")

    print("\n   [RK4]")
    print("   9. RK4_CLASSIC (Quy tắc 1/6)")
    print("   10. RK4_38 (Quy tắc 3/8)")

    print("\n   [RK5]")
    print("   11. RK5 (Butcher, 6 nấc)")

    while True:
        try:
            choice = int(input("\nChọn phương pháp (1-11): "))
            if 1 <= choice <= 11:
                break
            print("⚠️  Chọn từ 1 đến 11!")
        except:
            print("⚠️  Nhập số nguyên!")

    # Tạo Bảng Butcher
    if choice == 1:
        tableau = ButcherLibrary.get_method('RK1')
    elif choice == 2:
        tableau = ButcherLibrary.get_method('RK2_HEUN')
    elif choice == 3:
        tableau = ButcherLibrary.get_method('RK2_MIDPOINT')
    elif choice == 4:
        tableau = ButcherLibrary.get_method('RK2_RALSTON')
    elif choice == 5:
        alpha2 = float(input("\nNhập α₂ (khác 0): "))
        tableau = CustomButcherBuilder.build_rk2_custom(alpha2)
    elif choice == 6:
        tableau = ButcherLibrary.get_method('RK3_CLASSIC')
    elif choice == 7:
        tableau = ButcherLibrary.get_method('RK3_HEUN')
    elif choice == 8:
        alpha2 = float(input("\nNhập α₂: "))
        alpha3 = float(input("Nhập α₃: "))
        tableau = CustomButcherBuilder.build_rk3_custom(alpha2, alpha3)
    elif choice == 9:
        tableau = ButcherLibrary.get_method('RK4_CLASSIC')
    elif choice == 10:
        tableau = ButcherLibrary.get_method('RK4_38')
    else:  # choice == 11
        tableau = ButcherLibrary.get_method('RK5')

    print(f"\n✅ Đã chọn: {tableau.name}")
    print(f"   Số nấc (s): {tableau.s}")
    print(f"   Bậc (p): {tableau.p}")

    # Bước 2: Nhập phương trình
    print("\n" + "="*80)
    print("BƯỚC 2: ĐỊNH NGHĨA PHƯƠNG TRÌNH")
    print("="*80)

    f, f_expr, y_exact, has_exact = get_user_function()

    # Bước 3: Điều kiện đầu
    print("\n" + "="*80)
    print("BƯỚC 3: ĐIỀU KIỆN ĐẦU")
    print("="*80)

    x0, y0, x_end, h = get_initial_conditions()

    # Bước 4: Giải bài toán
    print("\n" + "="*80)
    print("BƯỚC 4: GIẢI BÀI TOÁN")
    print("="*80)

    solver = RungeKuttaSolver(tableau)

    print(f"\n🔄 Đang giải...")
    x_vals, y_vals = solver.solve(f, x0, y0, x_end, h)

    print(f"✅ Hoàn thành! Tính được {len(x_vals)} điểm")

    # Hiển thị kết quả
    print(f"\n📊 KẾT QUẢ:")
    print(f"\n{'Bước':<6} {'x':<12} {'y':<18}")
    print("-" * 40)

    # In một số bước đầu và cuối
    indices = list(range(min(5, len(x_vals)))) + [len(x_vals)-1]
    for i in indices:
        if i == 5:
            print("  ...   ...          ...")
        else:
            print(f"{i:<6} {x_vals[i]:<12.6f} {y_vals[i]:<18.10f}")

    # Vẽ đồ thị
    print(f"\n📈 VẼ ĐỒ THỊ...")

    fig, axes = plt.subplots(1, 2 if has_exact else 1, figsize=(14 if has_exact else 8, 6))

    if has_exact:
        ax1, ax2 = axes
    else:
        ax1 = axes

    # Đồ thị nghiệm
    ax1.plot(x_vals, y_vals, 'ro-', linewidth=2, markersize=5, label=f'{tableau.name}')

    if has_exact:
        x_smooth = np.linspace(x0, x_end, 500)
        y_smooth = y_exact(x_smooth)
        ax1.plot(x_smooth, y_smooth, 'b-', linewidth=2.5, alpha=0.7, label='Nghiệm chính xác')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(f'Nghiệm: y\' = {f_expr}, y({x0}) = {y0}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Đồ thị sai số (nếu có nghiệm chính xác)
    if has_exact:
        y_exact_vals = y_exact(x_vals)
        errors = np.abs(y_vals - y_exact_vals)

        ax2.semilogy(x_vals, errors, 'mo-', linewidth=2, markersize=5)
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('Sai số tuyệt đối', fontsize=12)
        ax2.set_title(f'Sai số (max: {np.max(errors):.6e})', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')

        print(f"\n📉 SAI SỐ:")
        print(f"   Sai số lớn nhất: {np.max(errors):.6e}")
        print(f"   Sai số trung bình: {np.mean(errors):.6e}")
        print(f"   Sai số tại x_end: {errors[-1]:.6e}")

    plt.tight_layout()

    # Lưu file
    method_name = tableau.name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    filename = f'/mnt/user-data/outputs/user_solution_{method_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Đã lưu đồ thị: {filename}")

    # Hỏi có muốn phân tích thêm không
    if has_exact:
        print("\n" + "="*80)
        print("PHÂN TÍCH THÊM")
        print("="*80)

        analyze = input("\nBạn có muốn:\n  1. Kiểm tra hội tụ\n  2. Vẽ miền ổn định\n  3. Cả hai\n  4. Không\nChọn (1-4): ")

        if analyze in ['1', '3']:
            print("\n🔬 KIỂM TRA HỘI TỤ...")
            results = RKAnalyzer.convergence_test(
                solver, f, y_exact, x0, y0, x_end,
                h_base=h, n_refinements=4
            )

        if analyze in ['2', '3']:
            print("\n🎨 VẼ MIỀN ỔN ĐỊNH...")
            fig = RKAnalyzer.plot_stability_region(solver)
            stability_file = f'/mnt/user-data/outputs/user_stability_{method_name}.png'
            fig.savefig(stability_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Đã lưu: {stability_file}")

    print("\n" + "="*80)
    print("HOÀN THÀNH!")
    print("="*80)


def main():
    """Hàm main - Chọn chế độ"""
    print("\n" + "="*80)
    print(" "*20 + "RUNGE-KUTTA SOLVER")
    print("="*80)

    print("\nChọn chế độ:")
    print("  1. Chế độ tương tác (tùy chỉnh phương pháp và phương trình)")
    print("  2. Chế độ demo (test tất cả phương pháp)")

    mode = input("\nChọn (1/2, mặc định 1): ").strip() or "1"

    if mode == "2":
        demo_comprehensive()
    else:
        interactive_mode()


def demo_comprehensive():
    """Demo toàn diện các tính năng (giữ nguyên code cũ)"""
    print("\n" + "="*80)
    print(" "*25 + "RUNGE-KUTTA SOLVER DEMO")
    print("="*80)

    # Bài toán test: y' = -y, y(0) = 1, nghiệm: y = e^(-x)
    f = lambda x, y: -y
    y_exact = lambda x: np.exp(-x)
    x0, y0, x_end = 0.0, 1.0, 2.0

    print(f"\n📌 BÀI TOÁN TEST:")
    print(f"   y' = -y")
    print(f"   y(0) = 1")
    print(f"   Nghiệm chính xác: y(x) = e^(-x)")
    print(f"   Khoảng: [{x0}, {x_end}]")

    methods = ['RK1', 'RK2_HEUN', 'RK2_MIDPOINT', 'RK2_RALSTON',
               'RK3_CLASSIC', 'RK3_HEUN', 'RK4_CLASSIC', 'RK4_38', 'RK5']

    convergence_results = {}

    for method_name in methods:
        print(f"\n{'='*80}")
        print(f"PHƯƠNG PHÁP: {method_name}")
        print(f"{'='*80}")

        tableau = ButcherLibrary.get_method(method_name)
        solver = RungeKuttaSolver(tableau)

        print(f"\n📊 BẢNG BUTCHER:")
        print(f"   Số nấc (s): {tableau.s}")
        print(f"   Bậc (p): {tableau.p}")
        print(f"   α = {tableau.alpha}")
        print(f"   r = {tableau.r}")

        results = RKAnalyzer.convergence_test(
            solver, f, y_exact, x0, y0, x_end,
            h_base=0.2, n_refinements=5
        )
        convergence_results[method_name] = results

        h = 0.1
        x_vals, y_vals = solver.solve(f, x0, y0, x_end, h)
        y_exact_vals = y_exact(x_vals)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x_vals, y_exact_vals, 'b-', linewidth=2.5, label='Nghiệm chính xác')
        plt.plot(x_vals, y_vals, 'ro--', linewidth=1.5, markersize=5, label=f'{method_name} (h={h})')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title(f'So sánh nghiệm - {method_name}', fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        errors = np.abs(y_vals - y_exact_vals)
        plt.semilogy(x_vals, errors, 'mo-', linewidth=2, markersize=5)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('Sai số tuyệt đối', fontsize=12)
        plt.title(f'Sai số - {method_name}', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(f'/mnt/user-data/outputs/rk_solution_{method_name}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✅ Đã lưu: rk_solution_{method_name}.png")

    print(f"\n{'='*80}")
    print("VẼ ĐỒ THỊ HỘI TỤ TỔNG HỢP")
    print(f"{'='*80}")

    fig = RKAnalyzer.plot_convergence(convergence_results)
    fig.savefig('/mnt/user-data/outputs/rk_convergence_all.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã lưu: rk_convergence_all.png")

    print(f"\n{'='*80}")
    print("VẼ MIỀN ỔN ĐỊNH")
    print(f"{'='*80}")

    stability_methods = ['RK2_RALSTON', 'RK3_CLASSIC', 'RK4_CLASSIC', 'RK5']

    for method_name in stability_methods:
        tableau = ButcherLibrary.get_method(method_name)
        solver = RungeKuttaSolver(tableau)

        fig = RKAnalyzer.plot_stability_region(solver)
        fig.savefig(f'/mnt/user-data/outputs/rk_stability_{method_name}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Đã lưu: rk_stability_{method_name}.png")

    print(f"\n{'='*80}")
    print("HOÀN THÀNH!")
    print(f"{'='*80}")
    print("\nTất cả file đã được lưu trong /mnt/user-data/outputs/")


if __name__ == "__main__":
    main()