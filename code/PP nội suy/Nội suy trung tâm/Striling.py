import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math

def get_float_input(prompt):
    """Hàm tiện ích để nhận input là số thực một cách an toàn."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Giá trị nhập không hợp lệ. Vui lòng nhập một số.")

def get_int_input(prompt):
    """Hàm tiện ích để nhận input là số nguyên một cách an toàn."""
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Giá trị nhập không hợp lệ. Vui lòng nhập một số nguyên.")

def build_diff_table(x, y, n):
    """Xây dựng và hiển thị bảng sai phân."""
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = diff_table[i + 1, j - 1] - diff_table[i, j - 1]

    print("\n## Bảng Sai Phân")
    print("x \t y \t", end="")
    for i in range(1, n):
        print(f"Δ^{i}y \t", end="")
    print("\n" + "-" * (8 * (n + 2)))

    for i in range(n):
        print(f"{x[i]}\t{y[i]}\t", end="")
        for j in range(1, n - i):
            print(f"{diff_table[i, j]:.4f}\t", end="")
        print()

    return diff_table

def stirling_interpolation():
    """Chương trình chính thực hiện nội suy Sterling."""

    print("--- CHƯƠNG TRÌNH NỘI SUY TRUNG TÂM STERLING ---")

    # --- Giai đoạn 1: Nhập dữ liệu ---
    precision = get_int_input("Nhập số chữ số sau dấu phẩy (sai số) mong muốn: ")

    n = get_int_input("Nhập số lượng mốc nội suy (n - phải là số lẻ): ")
    while n % 2 == 0:
        print("Lỗi: Công thức Sterling yêu cầu số mốc lẻ.")
        n = get_int_input("Nhập lại số mốc (n - phải là số lẻ): ")

    x_data = np.zeros(n)
    y_data = np.zeros(n)
    print("\nNhập các cặp giá trị (x, y):")
    for i in range(n):
        x_data[i] = get_float_input(f"Nhập x[{i}]: ")
        y_data[i] = get_float_input(f"Nhập y[{i}]: ")

    # Kiểm tra mốc cách đều
    h = x_data[1] - x_data[0]
    for i in range(2, n):
        if not np.isclose(x_data[i] - x_data[i-1], h):
            print(f"Lỗi: Các mốc không cách đều. (x[{i}] - x[{i-1}] != h)")
            return

    print(f"\nPhát hiện các mốc cách đều với h = {h}")

    # --- Giai đoạn 2: Xây dựng Bảng Sai Phân ---
    diff_table = build_diff_table(x_data, y_data, n)

    # --- Giai đoạn 3: Xây dựng Đa thức (Toán học) ---
    x0_index = n // 2
    x0 = x_data[x0_index]
    y0 = y_data[x0_index]

    t = sp.symbols('t')
    P_t = y0

    print(f"\n## Các bước xây dựng đa thức nội suy Sterling P(t)")
    print(f"Mốc trung tâm được chọn: x0 = {x0}, y0 = {y0}")
    print(f"Đặt t = (x - {x0}) / {h}")
    print(f"Công thức: P(t) = y0 + t * (Δy0 + Δy-1)/2 + (t^2/2!) * Δ^2y-1 + ...")
    print("-" * 30)
    print(f"P(t) = {y0} (Số hạng bậc 0)")

    term_odd_prod = t
    term_even_prod = t**2

    for i in range(1, n):
        if i % 2 == 1:  # Số hạng lẻ: 1, 3, 5, ...
            k = (i - 1) // 2
            diff1 = diff_table[x0_index - k - 1, i]
            diff2 = diff_table[x0_index - k, i]
            avg_diff = (diff1 + diff2) / 2

            term_symbolic = (term_odd_prod / math.factorial(i)) * avg_diff
            P_t += term_symbolic

            # Chuẩn bị cho số hạng chẵn tiếp theo
            if i > 1:
                term_even_prod *= (t**2 - k**2)

            print(f" + {term_symbolic} (Số hạng bậc {i})")

        else:  # Số hạng chẵn: 2, 4, 6, ...
            k = i // 2
            diff = diff_table[x0_index - k, i]

            term_symbolic = (term_even_prod / math.factorial(i)) * diff
            P_t += term_symbolic

            # Chuẩn bị cho số hạng lẻ tiếp theo
            term_odd_prod *= (t**2 - k**2)

            print(f" + {term_symbolic} (Số hạng bậc {i})")

    # --- Giai đoạn 4: Hiển thị Hàm số cụ thể ---
    print("\n" + "=" * 50)
    P_t_simplified = sp.expand(P_t)
    print(f"### Đa thức nội suy theo t (rút gọn):\nP(t) = {P_t_simplified}")

    x_sym = sp.symbols('x')
    t_sub = (x_sym - x0) / h

    P_x = P_t_simplified.subs(t, t_sub)
    P_x_simplified = sp.expand(P_x)
    print(f"\n### Đa thức nội suy theo x (Hàm số cụ thể):\nP(x) = {P_x_simplified}")
    print("=" * 50)

    # --- Giai đoạn 5: Vẽ Đồ thị ---
    print("\nĐang chuẩn bị đồ thị...")
    # Chuyển đổi hàm symbolic sympy sang hàm số numpy để tính toán
    f = sp.lambdify(x_sym, P_x_simplified, 'numpy')

    x_plot = np.linspace(np.min(x_data) - h, np.max(x_data) + h, 400)
    y_plot = f(x_plot)

    plt.figure(figsize=(12, 7))
    plt.plot(x_plot, y_plot, label="Đa thức nội suy P(x)", color='blue')
    plt.scatter(x_data, y_data, color='red', zorder=5, label="Các mốc nội suy")
    plt.title("Đồ thị hàm số nội suy Sterling")
    plt.xlabel("x")
    plt.ylabel("y = P(x)")
    plt.legend()
    plt.grid(True)
    plt.axvline(x=x0, color='gray', linestyle='--', label=f'Mốc trung tâm x0={x0}')
    plt.legend()
    print("Đã hiển thị đồ thị. Vui lòng kiểm tra cửa sổ mới.")
    plt.show()

    # --- Giai đoạn 6: Tính giá trị P(x) ---
    while True:
        choice = input("\nBạn có muốn tìm giá trị hàm số tại điểm x cụ thể không? (c/k): ").lower()
        if choice == 'k':
            break
        if choice == 'c':
            x_val = get_float_input("Nhập giá trị x cần tính: ")

            # Tính bằng P(x)
            y_val = P_x_simplified.subs(x_sym, x_val)

            # Hiển thị bước giải qua t
            t_val = (x_val - x0) / h
            y_val_t = P_t_simplified.subs(t, t_val)

            print(f"--- Tính toán P(x = {x_val}) ---")
            print(f"Bước 1: Tính t = ({x_val} - {x0}) / {h} = {t_val:.{precision}f}")
            print(f"Bước 2: Thay t = {t_val:.{precision}f} vào P(t)")
            print(f"Kết quả: P(x = {x_val}) = {y_val_t:.{precision}f}")
            # print(f"(Kiểm tra bằng P(x): {y_val:.{precision}f})") # Dùng để debug

    # --- Giai đoạn 7: Tính Đạo hàm ---
    while True:
        choice = input("\nBạn có muốn tính đạo hàm của hàm số không? (c/k): ").lower()
        if choice == 'k':
            break
        if choice == 'c':
            m = get_int_input(f"Nhập cấp đạo hàm (m >= 1 và m < {n}): ")
            if m < 1 or m >= n:
                print(f"Cấp đạo hàm không hợp lệ. Phải >= 1 và < số mốc ({n}).")
                continue

            print(f"\n## Tính toán Đạo hàm cấp {m}")
            P_deriv = P_x_simplified
            for i in range(m):
                P_deriv = sp.diff(P_deriv, x_sym)

            print(f"### Hàm số đạo hàm cấp {m}:\nP^({m})(x) = {P_deriv}")

            # Tính giá trị đạo hàm
            while True:
                deriv_choice = input(f"Bạn có muốn tính giá trị đạo hàm P^({m})(x) tại một điểm x không? (c/k): ").lower()
                if deriv_choice == 'k':
                    break
                if deriv_choice == 'c':
                    x_deriv_val = get_float_input(f"Nhập giá trị x để tính P^({m})(x): ")
                    deriv_val = P_deriv.subs(x_sym, x_deriv_val)
                    print(f"--- Kết quả ---")
                    print(f"P^({m})(x = {x_deriv_val}) = {deriv_val:.{precision}f}")

    print("\n--- Chương trình kết thúc ---")

# Chạy chương trình chính
if __name__ == "__main__":
    stirling_interpolation()