import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math

def print_difference_table(diff_table, x_values, n):
    """In Bảng Sai Phân một cách rõ ràng."""
    print("\n--- BẢNG SAI PHÂN HỮU HẠN ---")

    # In tiêu đề cột
    header = "x_i".ljust(10) + "y_i".ljust(10)
    for i in range(1, n):
        header += f"Δ^{i}y".ljust(10)
    print(header)
    print("-" * len(header))

    # In dữ liệu
    for i in range(n):
        line = f"{x_values[i]:<10.4f}{diff_table[i][0]:<10.4f}"
        for j in range(1, n - i):
            line += f"{diff_table[i][j]:<10.4f}"
        print(line)
    print("-" * len(header))

def get_user_input():
    """Thu thập và xác thực dữ liệu đầu vào từ người dùng."""

    print("--- CHƯƠNG TRÌNH NỘI SUY BESSEL ---")
    print("Lưu ý: Phương pháp này yêu cầu số mốc chẵn và các mốc x phải cách đều.")

    while True:
        try:
            n = int(input("Nhập số lượng mốc dữ liệu (phải là số chẵn > 0): "))
            # Điều kiện số mốc chẵn
            if n > 0 and n % 2 == 0:
                break
            else:
                print("Lỗi: Số lượng mốc phải là số chẵn và lớn hơn 0.")
        except ValueError:
            print("Lỗi: Vui lòng nhập một số nguyên.")

    x_values = []
    y_values = []
    print("\nNhập các cặp giá trị (x, y):")
    for i in range(n):
        while True:
            try:
                x_val = float(input(f"  Nhập x[{i}]: "))
                y_val = float(input(f"  Nhập y[{i}]: "))
                x_values.append(x_val)
                y_values.append(y_val)
                break
            except ValueError:
                print("Lỗi: Vui lòng nhập giá trị số.")

    # Kiểm tra điều kiện mốc cách đều [cite: 261]
    h = x_values[1] - x_values[0]
    is_equally_spaced = True
    for i in range(2, n):
        # Sử dụng so sánh gần đúng để tránh lỗi float
        if not math.isclose(x_values[i] - x_values[i-1], h):
            is_equally_spaced = False
            break

    if not is_equally_spaced:
        print(f"\nLỗi: Các mốc x không cách đều. (h={h} không nhất quán).")
        print("Phương pháp Bessel không thể áp dụng. Vui lòng khởi động lại.")
        return None

    print(f"\nPhát hiện các mốc cách đều với h = {h}")

    while True:
        try:
            precision = int(input("Nhập số chữ số sau dấu phẩy cho kết quả: "))
            if precision >= 0:
                break
            else:
                print("Lỗi: Vui lòng nhập một số không âm.")
        except ValueError:
            print("Lỗi: Vui lòng nhập một số nguyên.")

    return n, np.array(x_values), np.array(y_values), h, precision

def build_difference_table(y_values, n):
    """Xây dựng bảng sai phân từ các giá trị y."""
    # Khởi tạo bảng (n x n) với các giá trị 0.0
    diff_table = np.zeros((n, n))

    # Cột đầu tiên là giá trị y
    diff_table[:, 0] = y_values

    # Tính toán các cột sai phân
    for j in range(1, n):  # Cột (cấp sai phân)
        for i in range(n - j): # Hàng
            diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]

    return diff_table

def run_bessel_interpolation():
    """Hàm chính thực hiện toàn bộ quy trình."""

    # 1. Thu thập dữ liệu
    inputs = get_user_input()
    if inputs is None:
        return
    n, x_values, y_values, h, precision = inputs

    # 2. Xây dựng và in bảng sai phân [cite: 262]
    diff_table = build_difference_table(y_values, n)
    print_difference_table(diff_table, x_values, n)

    # 3. Xây dựng đa thức nội suy
    print("\n--- XÂY DỰNG ĐA THỨC NỘI SUY BESSEL ---")

    # Khởi tạo biến tượng trưng
    t = sp.Symbol('t')
    x = sp.Symbol('x')

    # Xác định mốc trung tâm
    origin_index = n // 2 - 1
    x0_val = x_values[origin_index]
    y0_val = y_values[origin_index]
    y1_val = y_values[origin_index + 1]

    print(f"Số mốc n = {n} (chẵn).")
    print(f"Mốc trung tâm được chọn là x_{origin_index} = {x0_val} và x_{origin_index+1} = {y_values[origin_index+1]}")
    print(f"Sử dụng công thức t = (x - x0) / h = (x - {x0_val}) / {h}")

    # Áp dụng công thức Bessel
    # P(t) = (y0 + y1)/2 + (t - 1/2)Δy0 + [t(t-1)/2!] * (Δ²y-1 + Δ²y0)/2 + ...

    # Khởi tạo đa thức
    P_t = (y0_val + y1_val) / 2

    # Chuỗi hiển thị các bước giải
    steps_str = f"P(t) = ({y0_val} + {y1_val}) / 2 \n"

    # Các biến để xây dựng đa thức
    term_poly_odd = (t - 0.5)
    term_poly_even = 1

    # Term 1 (i=1): (t - 1/2) * Δy0
    val_delta_y0 = diff_table[origin_index][1]
    P_t += term_poly_odd * val_delta_y0
    steps_str += f"     + ({val_delta_y0:.{precision}f}) * ({term_poly_odd}) \n"

    # Vòng lặp cho các số hạng bậc cao (từ i=2 đến n-1)
    for i in range(2, n):
        k = i // 2

        if i % 2 == 0: # Bậc chẵn (i=2, 4, 6, ...)
            # (Δ²y-1 + Δ²y0) / (2 * 2!)
            idx = origin_index - k + 1
            if idx < 0 or (idx + k) >= n: break # Ra ngoài bảng

            val_delta_1 = diff_table[idx-1][i]
            val_delta_2 = diff_table[idx][i]
            coeff = (val_delta_1 + val_delta_2) / (2 * math.factorial(i))

            # t(t-1)
            if i == 2:
                term_poly_even = t * (t - 1)
            else:
                term_poly_even *= (t + k - 1) * (t - k)

            P_t += coeff * term_poly_even
            steps_str += f"     + ({coeff:.{precision}f}) * ({term_poly_even}) \n"

        else: # Bậc lẻ (i=3, 5, 7, ...)
            # Δ³y-1 / 3!
            idx = origin_index - k
            if idx < 0 or (idx + k + 1) >= n: break # Ra ngoài bảng

            val_delta = diff_table[idx][i]
            coeff = val_delta / math.factorial(i)

            # (t-1/2) * t * (t-1)
            term_poly_odd *= (t + k - 1) * (t - k)

            P_t += coeff * term_poly_odd
            steps_str += f"     + ({coeff:.{precision}f}) * ({term_poly_odd}) \n"

    print("\nCác bước xây dựng P(t):")
    print(steps_str)

    # Rút gọn P(t)
    P_t_simplified = sp.expand(P_t)
    print(f"\nĐa thức P(t) rút gọn:\nP(t) = {P_t_simplified}")

    # 4. Chuyển đổi P(t) -> P(x)
    P_x = P_t.subs(t, (x - x0_val) / h)
    P_x_expanded = sp.expand(P_x)

    print("\n--- ĐA THỨC NỘI SUY P(x) ---")
    print(f"Thay t = (x - {x0_val}) / {h} và rút gọn, ta được:")
    print(f"\nP(x) = {P_x_expanded}\n")

    # 5. Vẽ đồ thị
    print("--- ĐANG VẼ ĐỒ THỊ ---")
    # Chuyển đổi hàm P(x) tượng trưng sang hàm số có thể tính toán
    P_func = sp.lambdify(x, P_x_expanded, 'numpy')

    # Tạo các điểm để vẽ
    x_plot = np.linspace(min(x_values) - h, max(x_values) + h, 400)
    y_plot = P_func(x_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, label=f'Đa thức P(x)', color='blue')
    plt.scatter(x_values, y_values, color='red', zorder=5, label='Dữ liệu gốc')
    plt.title('Đồ thị Nội suy Bessel')
    plt.xlabel('x')
    plt.ylabel('y = P(x)')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=x0_val, color='gray', linestyle='--', label=f'x0 = {x0_val}')
    plt.legend()
    print("Đã tạo đồ thị. Vui lòng kiểm tra cửa sổ mới...")
    plt.show()

    # 6. Tính giá trị P(x) tại điểm x cụ thể
    while True:
        choice = input("\nBạn có muốn tính giá trị hàm số tại 1 điểm x cụ thể? (c/k): ").lower()
        if choice == 'c':
            try:
                x_eval = float(input("  Nhập giá trị x cần tính: "))
                # Tính giá trị bằng cách thay x vào P(x)
                y_eval = P_x_expanded.subs(x, x_eval).evalf()
                print(f"  -> P({x_eval}) = {y_eval:.{precision}f}")
            except ValueError:
                print("Lỗi: Vui lòng nhập một số.")
        elif choice == 'k':
            break

    # 7. Tính đạo hàm
    while True:
        choice = input("\nBạn có muốn tính đạo hàm cấp m của P(x)? (c/k): ").lower()
        if choice == 'c':
            try:
                m = int(input(f"  Nhập cấp đạo hàm m (m >= 0): "))
                if m < 0:
                    print("Lỗi: Cấp đạo hàm phải >= 0.")
                    continue

                # Tính đạo hàm
                P_deriv = P_x_expanded
                for i in range(m):
                    P_deriv = sp.diff(P_deriv, x)

                print(f"\n--- ĐẠO HÀM CẤP {m} ---")
                print(f"P^({m})(x) = {P_deriv}")

                # 8. Tính giá trị đạo hàm tại 1 điểm
                while True:
                    eval_choice = input(f"\n  Bạn có muốn tính giá trị của P^({m})(x) tại 1 điểm x? (c/k): ").lower()
                    if eval_choice == 'c':
                        try:
                            x_deriv_eval = float(input(f"    Nhập giá trị x để tính P^({m})(x): "))
                            y_deriv_eval = P_deriv.subs(x, x_deriv_eval).evalf()
                            print(f"    -> P^({m})({x_deriv_eval}) = {y_deriv_eval:.{precision}f}")
                        except ValueError:
                            print("Lỗi: Vui lòng nhập một số.")
                    elif eval_choice == 'k':
                        break

            except ValueError:
                print("Lỗi: Vui lòng nhập một số nguyên.")
        elif choice == 'k':
            break

    print("\n--- KẾT THÚC CHƯƠNG TRÌNH ---")

# Chạy chương trình chính
if __name__ == "__main__":
    run_bessel_interpolation()