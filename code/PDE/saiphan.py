import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class FiniteDifferenceSolver:
    def __init__(self):
        self.a = None
        self.b = None
        self.h = None
        self.N = None
        self.x = None
        self.p_func = None
        self.q_func = None
        self.f_func = None
        self.bc_a = None
        self.bc_b = None
        self.A = None
        self.b = None
        self.u = None

    def input_interval(self):
        print("=== Bước 1: Nhập khoảng [a, b] và bước lưới h ===")
        self.a = float(input("Nhập a: "))
        self.b = float(input("Nhập b: "))
        self.h = float(input("Nhập bước lưới h: "))
        self.N = int((self.b - self.a) / self.h)
        self.x = np.linspace(self.a, self.b, self.N + 1)
        print(f"Số điểm lưới N = {self.N}")
        print(f"Các điểm lưới x_i: {self.x}\n")

    def input_functions(self):
        print("=== Bước 2: Nhập các hàm p(x), q(x), f(x) ===")
        print("Lưu ý: Sử dụng cú pháp Python, ví dụ: x**2 + 1, 4*x**4 - 4*x**2 + 5, 1.25*x*exp(-x**2)")
        p_expr = input("Nhập p(x): ")
        q_expr = input("Nhập q(x): ")
        f_expr = input("Nhập f(x): ")
        x_sym = sp.symbols('x')
        self.p_func = sp.lambdify(x_sym, sp.sympify(p_expr), 'numpy')
        self.q_func = sp.lambdify(x_sym, sp.sympify(q_expr), 'numpy')
        self.f_func = sp.lambdify(x_sym, sp.sympify(f_expr), 'numpy')
        print("Đã nhập các hàm.\n")

    def input_boundary_conditions(self):
        print("=== Bước 3: Nhập điều kiện biên ===")
        print("Tại x = a:")
        self.bc_a = self._input_single_bc("a")
        print("Tại x = b:")
        self.bc_b = self._input_single_bc("b")
        print()

    def _input_single_bc(self, endpoint):
        bc = {}
        print(f"Chọn loại điều kiện biên tại {endpoint}:")
        print("1. Dirichlet (cho giá trị hàm)")
        print("2. Neumann (cho đạo hàm p(x)u'(x))")
        print("3. Robin (hỗn hợp: p(x)u'(x) - σ*u(x))")
        choice = int(input("Lựa chọn (1/2/3): "))
        bc['type'] = choice
        if choice == 1:
            bc['value'] = float(input(f"Nhập u({endpoint}): "))
        elif choice == 2:
            bc['value'] = float(input(f"Nhập p({endpoint}) * u'({endpoint}): "))
        else:
            sigma = float(input(f"Nhập σ (sigma) cho điều kiện Robin: "))
            bc['sigma'] = sigma
            bc['value'] = float(input(f"Nhập giá trị vế phải (p({endpoint})*u'({endpoint}) - σ*u({endpoint})): "))
        return bc

    def build_system(self):
        print("=== Bước 4: Xây dựng hệ phương trình sai phân ===")
        N = self.N
        self.A = np.zeros((N + 1, N + 1))
        self.b = np.zeros(N + 1)

        # Điều kiện biên tại a
        if self.bc_a['type'] == 1:  # Dirichlet
            self.A[0, 0] = 1
            self.b[0] = self.bc_a['value']
        else:
            x0 = self.x[0]
            p_half = self.p_func(x0 + self.h/2)
            q0 = self.q_func(x0)
            f0 = self.f_func(x0)
            if self.bc_a['type'] == 2:  # Neumann
                mu = self.bc_a['value']
                self.A[0, 0] = -(p_half + (self.h**2 / 2) * q0)
                self.A[0, 1] = p_half
                self.b[0] = (self.h**2 / 2) * f0 - mu * self.h
            else:  # Robin
                sigma = self.bc_a['sigma']
                mu = self.bc_a['value']
                self.A[0, 0] = -(p_half + (self.h**2 / 2) * q0 + sigma * self.h)
                self.A[0, 1] = p_half
                self.b[0] = (self.h**2 / 2) * f0 - mu * self.h

        # Phương trình tại các điểm trong (i = 1 đến N-1)
        for i in range(1, N):
            xi = self.x[i]
            p_plus = self.p_func(xi + self.h/2)
            p_minus = self.p_func(xi - self.h/2)
            qi = self.q_func(xi)
            fi = self.f_func(xi)
            self.A[i, i-1] = p_minus
            self.A[i, i] = -(p_plus + p_minus + self.h**2 * qi)
            self.A[i, i+1] = p_plus
            self.b[i] = -self.h**2 * fi

        # Điều kiện biên tại b
        if self.bc_b['type'] == 1:  # Dirichlet
            self.A[N, N] = 1
            self.b[N] = self.bc_b['value']
        else:
            xN = self.x[N]
            p_half = self.p_func(xN - self.h/2)
            qN = self.q_func(xN)
            fN = self.f_func(xN)
            if self.bc_b['type'] == 2:  # Neumann
                mu = self.bc_b['value']
                self.A[N, N-1] = -p_half
                self.A[N, N] = p_half + (self.h**2 / 2) * qN
                self.b[N] = (self.h**2 / 2) * fN - mu * self.h
            else:  # Robin
                sigma = self.bc_b['sigma']
                mu = self.bc_b['value']
                self.A[N, N-1] = -p_half
                self.A[N, N] = p_half + (self.h**2 / 2) * qN - sigma * self.h
                self.b[N] = (self.h**2 / 2) * fN - mu * self.h

        print("Ma trận hệ số A:")
        print(self.A)
        print("\nVector vế phải b:")
        print(self.b)
        print()

    def solve(self):
        print("=== Bước 5: Giải hệ phương trình ===")
        self.u = np.linalg.solve(self.A, self.b)
        print("Nghiệm xấp xỉ u_i:")
        for i, ui in enumerate(self.u):
            print(f"u[{i}] = {ui:.6f} tại x = {self.x[i]:.2f}")
        print()

    def plot(self):
        print("=== Bước 6: Vẽ đồ thị nghiệm ===")
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.u, 'b-o', linewidth=2, markersize=4, label='Nghiệm xấp xỉ')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Nghiệm của bài toán biên bằng phương pháp sai phân hữu hạn')
        plt.grid(True)
        plt.legend()
        plt.show()

    def run(self):
        print("Chương trình giải bài toán biên bằng phương pháp sai phân hữu hạn")
        print("Phương trình: [p(x) u'(x)]' - q(x) u(x) = -f(x)\n")
        self.input_interval()
        self.input_functions()
        self.input_boundary_conditions()
        self.build_system()
        self.solve()
        self.plot()

# Chạy chương trình
if __name__ == "__main__":
    solver = FiniteDifferenceSolver()
    solver.run()