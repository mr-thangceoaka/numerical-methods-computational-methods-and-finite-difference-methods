import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols

# =============================
# BẢNG TỶ SAI PHÂN & SAI PHÂN
# =============================
def divided_differences(x, y):
    """Tính bảng tỷ sai phân (Newton tổng quát)."""
    n = len(y)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
    return table

def forward_differences(y):
    """Bảng sai phân tiến Δ^k y_i."""
    n = len(y)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]
    return table

def backward_differences(y):
    """Bảng sai phân lùi ∇^k y_i."""
    n = len(y)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            table[i][j] = table[i][j - 1] - table[i - 1][j - 1]
    return table

# =============================
# XÂY DỰNG ĐA THỨC NEWTON
# =============================
def newton_forward_polynomial(x_data, y_data, decimals):
    """Xây dựng đa thức Newton tiến (mốc cách đều)."""
    n = len(x_data)
    h = x_data[1] - x_data[0]
    # kiểm tra cách đều
    for i in range(1, n - 1):
        if abs((x_data[i + 1] - x_data[i]) - h) > 1e-10:
            raise ValueError("Các mốc không cách đều! Không thể sử dụng Newton tiến/lùi")

    fd_table = forward_differences(y_data)
    x = sp.Symbol('x')
    t = (x - x_data[0]) / h
    P = sp.Integer(0) + fd_table[0][0]

    print(f"\n=== XÂY DỰNG ĐA THỨC NEWTON TIẾN ===")
    print(f"Bước 0: P₀(x) = {round(fd_table[0][0], decimals)}")
    print(f"Khoảng cách h = {h}")
    print(f"Biến t = (x - {x_data[0]}) / {h}")

    t_product = 1
    poly_terms = [f"{round(fd_table[0][0], decimals)}"]

    for i in range(1, n):
        coef = fd_table[0][i] / sp.factorial(i)
        t_product *= (t - (i - 1))
        term = coef * t_product
        P += term

        print(f"\nBước {i}:")
        print(f"  Δ^{i}y₀ = {round(fd_table[0][i], decimals)}")
        print(f"  Hệ số: Δ^{i}y₀/{i}! = {round(coef, decimals)}")
        print(f"  Tích: t(t-1)...(t-{i-1}) = {t_product}")
        print(f"  P_{i}(x) = {sp.expand(P)}")

        poly_terms.append(f"{round(coef, decimals)} × {t_product}")

    return P, poly_terms, fd_table

def newton_backward_polynomial(x_data, y_data, decimals):
    """Xây dựng đa thức Newton lùi (mốc cách đều)."""
    n = len(x_data)
    h = x_data[1] - x_data[0]
    # kiểm tra cách đều
    for i in range(1, n - 1):
        if abs((x_data[i + 1] - x_data[i]) - h) > 1e-10:
            raise ValueError("Các mốc không cách đều! Không thể sử dụng Newton tiến/lùi")

    bd_table = backward_differences(y_data)
    x = sp.Symbol('x')
    t = (x - x_data[-1]) / h
    P = sp.Integer(0) + bd_table[-1][0]

    print(f"\n=== XÂY DỰNG ĐA THỨC NEWTON LÙI ===")
    print(f"Bước 0: P₀(x) = {round(bd_table[-1][0], decimals)}")
    print(f"Khoảng cách h = {h}")
    print(f"Biến t = (x - {x_data[-1]}) / {h}")

    t_product = 1
    poly_terms = [f"{round(bd_table[-1][0], decimals)}"]

    for i in range(1, n):
        coef = bd_table[-1][i] / sp.factorial(i)
        t_product *= (t + (i - 1))
        term = coef * t_product
        P += term

        print(f"\nBước {i}:")
        print(f"  ∇^{i}y_{n-1} = {round(bd_table[-1][i], decimals)}")
        print(f"  Hệ số: ∇^{i}y_{n-1}/{i}! = {round(coef, decimals)}")
        print(f"  Tích: t(t+1)...(t+{i-1}) = {t_product}")
        print(f"  P_{i}(x) = {sp.expand(P)}")

        poly_terms.append(f"{round(coef, decimals)} × {t_product}")

    return P, poly_terms, bd_table

def newton_general_polynomial(x_data, table, decimals):
    """Xây dựng đa thức Newton tổng quát (mốc bất kỳ)."""
    x = sp.Symbol('x')
    P = sp.Integer(0) + table[0][0]

    print(f"\n=== XÂY DỰNG ĐA THỨC NEWTON TỔNG QUÁT ===")
    print(f"Bước 0: P₀(x) = {round(table[0][0], decimals)}")

    poly_terms = [f"{round(table[0][0], decimals)}"]

    for i in range(1, len(x_data)):
        coef = round(table[0][i], decimals)
        product_term = 1
        product_str = ""
        for j in range(i):
            product_term *= (x - x_data[j])
            if j == 0:
                product_str += f"(x - {round(x_data[j], decimals)})"
            else:
                product_str += f" × (x - {round(x_data[j], decimals)})"

        term = coef * product_term
        P += term

        print(f"\nBước {i}:")
        print(f"  f[x₀,...,x_{i}] = {coef}")
        print(f"  Thành phần tích: {product_str}")
        print(f"  P_{i}(x) = {sp.expand(P)}")

        poly_terms.append(f"{coef} × {product_str}")

    return P, poly_terms

# =============================
# IN CÁC BƯỚC TỶ SAI PHÂN
# =============================
def print_detailed_steps(x_data, y_data, table, decimals):
    print(f"\n=== TÍNH TOÁN TỶ SAI PHÂN CHI TIẾT ===")
    n = len(x_data)

    print("Tỷ sai phân cấp 0 (giá trị hàm):")
    for i in range(n):
        print(f"  f[x_{i}] = f({round(x_data[i], decimals)}) = {round(y_data[i], decimals)}")

    print("\nTỷ sai phân cấp 1:")
    for i in range(n - 1):
        numerator = round(y_data[i + 1] - y_data[i], decimals)
        denominator = round(x_data[i + 1] - x_data[i], decimals)
        result = round(table[i][1], decimals)
        print(f"  f[x_{i}, x_{i+1}] = [{round(y_data[i+1], decimals)} - {round(y_data[i], decimals)}] / "
              f"[{round(x_data[i+1], decimals)} - {round(x_data[i], decimals)}] = {numerator} / {denominator} = {result}")

    for j in range(2, n):
        print(f"\nTỷ sai phân cấp {j}:")
        for i in range(n - j):
            num = round(table[i + 1][j - 1] - table[i][j - 1], decimals)
            denom = round(x_data[i + j] - x_data[i], decimals)
            result = round(table[i][j], decimals)
            print(f"  f[x_{i},...,x_{i+j}] = [{round(table[i+1][j-1], decimals)} - "
                  f"{round(table[i][j-1], decimals)}] / [{round(x_data[i+j], decimals)} - "
                  f"{round(x_data[i], decimals)}] = {num} / {denom} = {result}")

# =============================
# VẼ ĐỒ THỊ
# =============================
def plot_polynomial(P, x_data, y_data, title, decimals):
    """Vẽ đồ thị đa thức đã nội suy (không đặt màu cụ thể)."""
    x_plot = np.linspace(min(x_data) - 0.5, max(x_data) + 0.5, 400)
    P_func = sp.lambdify(sp.Symbol('x'), P, 'numpy')
    y_plot = P_func(x_plot)

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_plot, linewidth=2, label='Đa thức nội suy')
    plt.scatter(x_data, y_data, s=40, label='Mốc nội suy')

    for xi, yi in zip(x_data, y_data):
        plt.annotate(f'({round(xi, decimals)}, {round(yi, decimals)})',
                     (xi, yi), xytext=(5, 5), textcoords='offset points')

    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================
# ĐÁNH GIÁ f(x) VỚI CẤP m
# =============================
def evaluate_with_given_order(x_data, y_data, x_target, m, decimals):
    """
    Xấp xỉ f(x_target) bằng công thức Newton tiến/lùi với
    SAI PHÂN TỚI CẤP m (0 <= m <= n-1).
    Trả về: (value, method, m_used, contributions, used_diffs_row, h, t)
    """
    n = len(x_data)
    if n == 1:
        return y_data[0], "TRIVIAL", 0, [y_data[0]], None, None, None

    # Kiểm tra mốc cách đều
    h = x_data[1] - x_data[0]
    for i in range(2, n):
        if abs((x_data[i] - x_data[i-1]) - h) > 1e-10:
            raise ValueError("Mốc không cách đều – không dùng được sai phân tiến/lùi cấp m.")

    # Chuẩn hoá m
    if m < 0:
        m = 0
    if m > n - 1:
        m = n - 1

    # Chọn tiến / lùi theo vị trí x_target
    use_forward = abs(x_target - x_data[0]) <= abs(x_target - x_data[-1])
    contributions = []

    if use_forward:
        fd_table = forward_differences(y_data)
        t = (x_target - x_data[0]) / h

        # P_m(x) = y0 + Δy0 t + Δ^2 y0 / 2! t(t-1) + ... tới cấp m
        value = float(fd_table[0][0])
        contributions.append(value)
        t_prod = 1.0

        for i in range(1, m + 1):
            t_prod *= (t - (i - 1))
            term = float(fd_table[0][i]) / float(sp.factorial(i)) * float(t_prod)
            contributions.append(term)
            value += term

        used_diffs_row = fd_table[0, :m + 1]
        method = "NEWTON TIẾN"
        return value, method, m, contributions, used_diffs_row, h, t

    else:
        bd_table = backward_differences(y_data)
        t = (x_target - x_data[-1]) / h

        # P_m(x) = y_{n-1} + ∇y_{n-1} t + ∇^2 y_{n-1} / 2! t(t+1) + ... tới cấp m
        value = float(bd_table[-1][0])
        contributions.append(value)
        t_prod = 1.0

        for i in range(1, m + 1):
            t_prod *= (t + (i - 1))
            term = float(bd_table[-1][i]) / float(sp.factorial(i)) * float(t_prod)
            contributions.append(term)
            value += term

        used_diffs_row = bd_table[-1, :m + 1]
        method = "NEWTON LÙI"
        return value, method, m, contributions, used_diffs_row, h, t

# =============================
# CHƯƠNG TRÌNH CHÍNH
# =============================
def main():
    print("=== CHƯƠNG TRÌNH NỘI SUY NEWTON CHI TIẾT ===")

    n = int(input("Nhập số lượng điểm nội suy: "))
    x_data, y_data = [], []

    print("Nhập các điểm (x, y):")
    for i in range(n):
        x = float(input(f"x[{i}] = "))
        y = float(input(f"y[{i}] = "))
        x_data.append(x)
        y_data.append(y)

    # kiểm tra trùng mốc
    if len(x_data) != len(set(x_data)):
        print("LỖI: Các mốc nội suy phải khác nhau!")
        return

    decimals = int(input("Nhập số chữ số thập phân làm tròn: "))

    # kiểm tra cách đều
    is_equidistant = True
    if n > 1:
        h = x_data[1] - x_data[0]
        for i in range(2, n):
            if abs((x_data[i] - x_data[i - 1]) - h) > 1e-10:
                is_equidistant = False
                break

    # chọn phương pháp
    if is_equidistant and n > 1:
        print(f"\nCác mốc cách đều với h = {h}")
        print("Chọn phương pháp:")
        print("1. Newton tổng quát (tỷ sai phân)")
        print("2. Newton tiến")
        print("3. Newton lùi")
        choice = input("Nhập lựa chọn (1/2/3): ").strip()
    else:
        print("\nCác mốc không cách đều, sử dụng Newton tổng quát")
        choice = "1"

    # bảng tỷ sai phân
    table = divided_differences(x_data, y_data)

    # in bảng tỷ sai phân
    print("\n=== BẢNG TỶ SAI PHÂN HOÀN CHỈNH ===")
    print("x_i\tf[x_i]\t", end="")
    for i in range(1, n):
        print(f"f[...] cấp {i}\t", end="")
    print()

    for i in range(n):
        print(f"{round(x_data[i], decimals)}\t", end="")
        for j in range(n - i):
            print(f"{round(table[i][j], decimals)}\t", end="")
        print()

    # chi tiết các bước tỷ sai phân
    print_detailed_steps(x_data, y_data, table, decimals)

    # xây đa thức theo phương pháp đã chọn
    if choice == "2":
        try:
            P, poly_terms, fd_table = newton_forward_polynomial(x_data, y_data, decimals)
            method_name = "NEWTON TIẾN"

            print(f"\n=== BẢNG SAI PHÂN TIẾN ===")
            for i in range(n):
                print(f"y_{i}\t", end="")
                for j in range(n - i):
                    print(f"{round(fd_table[i][j], decimals)}\t", end="")
                print()

        except ValueError as e:
            print(f"Lỗi: {e}")
            print("Chuyển sang sử dụng Newton tổng quát")
            P, poly_terms = newton_general_polynomial(x_data, table, decimals)
            method_name = "NEWTON TỔNG QUÁT"

    elif choice == "3":
        try:
            P, poly_terms, bd_table = newton_backward_polynomial(x_data, y_data, decimals)
            method_name = "NEWTON LÙI"

            print(f"\n=== BẢNG SAI PHÂN LÙI ===")
            for i in range(n):
                print(f"y_{i}\t", end="")
                for j in range(n):
                    if j <= i:
                        print(f"{round(bd_table[i][j], decimals)}\t", end="")
                    else:
                        print("\t", end="")
                print()

        except ValueError as e:
            print(f"Lỗi: {e}")
            print("Chuyển sang sử dụng Newton tổng quát")
            P, poly_terms = newton_general_polynomial(x_data, table, decimals)
            method_name = "NEWTON TỔNG QUÁT"

    else:
        P, poly_terms = newton_general_polynomial(x_data, table, decimals)
        method_name = "NEWTON TỔNG QUÁT"

    # in đa thức cuối cùng
    print(f"\n=== ĐA THỨC NỘI SUY {method_name} HOÀN CHỈNH ===")
    print("Dạng tích:")
    poly_str = " + ".join(poly_terms)
    print(f"P(x) = {poly_str}")

    print("\nDạng khai triển:")
    P_expanded = sp.expand(P)
    print(f"P(x) = {P_expanded}")

    print("\nDạng thu gọn:")
    P_simplified = sp.simplify(P_expanded)
    print(f"P(x) = {P_simplified}")

    # vẽ đồ thị
    plot_polynomial(P, x_data, y_data, f'ĐỒ THỊ ĐA THỨC NỘI SUY {method_name}', decimals)

    # ===========================
    # ĐÁNH GIÁ GIÁ TRỊ / SAI PHÂN / ĐẠO HÀM
    # ===========================
    x_sym = sp.Symbol('x')

    while True:
        print("\n=== MENU ĐÁNH GIÁ ===")
        print("1. Tính giá trị đa thức P(x)")
        print("2. Xấp xỉ f(x) bằng công thức Newton với sai phân tới cấp m")
        print("3. Tính đạo hàm cấp m của P tại x")
        print("Nhập 'exit' để thoát.")
        mode = input("Lựa chọn của bạn (1/2/3/exit): ").strip()

        if mode.lower() == "exit":
            break

        try:
            x_target = float(input("Nhập giá trị x: "))
        except ValueError:
            print("x không hợp lệ.")
            continue

        # 1) Giá trị đa thức P(x)
        if mode == "1":
            P_func = sp.lambdify(x_sym, P, 'numpy')
            result = float(P_func(x_target))
            print(f"\nP({x_target}) = {result}")
            print(f"P({x_target}) ≈ {round(result, decimals)}")

        # 2) Giá trị gần đúng dùng sai phân tới cấp m
        elif mode == "2":
            try:
                m = int(input("Nhập cấp sai phân m (0 .. n-1): "))
            except ValueError:
                print("m không hợp lệ.")
                continue

            try:
                approx, meth, m_used, contr, used_row, h_val, t_val = evaluate_with_given_order(
                    x_data, y_data, x_target, m, decimals
                )
            except ValueError as e:
                print(f"Lỗi: {e}")
                # fallback: dùng đa thức tổng quát
                P_func = sp.lambdify(x_sym, P, 'numpy')
                result = float(P_func(x_target))
                print(f"\n(Mốc không cách đều) Dùng đa thức P(x):")
                print(f"P({x_target}) = {result}")
                print(f"P({x_target}) ≈ {round(result, decimals)}")
                continue

            anchor = x_data[0] if meth == "NEWTON TIẾN" else x_data[-1]
            print(f"\nPhương pháp: {meth}")
            print(f"h = {x_data[1]-x_data[0]},  t = (x - {anchor})/h = {t_val}")
            print(f"Dùng tới SAI PHÂN CẤP m = {m_used}")
            print(f"Giá trị xấp xỉ f({x_target}) ≈ {approx}")
            print(f"Làm tròn {decimals} chữ số: ≈ {round(approx, decimals)}")

            # In các sai phân đã dùng
            if meth == "NEWTON TIẾN":
                print("\nCác sai phân tiến Δ^i y0 đã dùng:")
            else:
                print("\nCác sai phân lùi ∇^i y_{n-1} đã dùng:")

            for i, val in enumerate(used_row):
                print(f"  Cấp {i}: {round(float(val), decimals)}")

            print("\nĐóng góp từng hạng:")
            for i, term in enumerate(contr):
                label = "hệ số tự do" if i == 0 else f"hạng cấp {i}"
                print(f"  {label}: {round(float(term), decimals)}")

        # 3) Đạo hàm cấp m của P tại x
        elif mode == "3":
            try:
                m = int(input("Nhập cấp đạo hàm m (0,1,2,...): "))
            except ValueError:
                print("m không hợp lệ.")
                continue

            if m < 0:
                print("m phải ≥ 0.")
                continue

            P_diff = sp.diff(P, x_sym, m)
            P_diff_func = sp.lambdify(x_sym, P_diff, 'numpy')
            val = float(P_diff_func(x_target))

            print(f"\nP^{m}(x) = {P_diff}")
            print(f"P^{m}({x_target}) = {val}")
            print(f"P^{m}({x_target}) ≈ {round(val, decimals)}")

        else:
            print("Lựa chọn không hợp lệ, hãy chọn 1/2/3 hoặc 'exit'.")

if __name__ == "__main__":
    main()
