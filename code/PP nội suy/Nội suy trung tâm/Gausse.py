import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, factorial, simplify, diff, lambdify
import math

# --- 1. HÀM TẠO BẢNG SAI PHÂN ---
def create_diff_table(y_values, n):
    """
    Tạo bảng sai phân hữu hạn.
    Bảng[i][j] sẽ lưu trữ Delta^j(y_i)
    """
    table = np.zeros((n, n))
    table[:, 0] = y_values

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

    return table

def print_diff_table(x_data, table, n):
    """In bảng sai phân ra console một cách dễ đọc."""
    print("\n--- BẢNG SAI PHÂN HỮU HẠN ---")
    header = "x \t y \t"
    header += "\t".join([f"Δ^{i}y" for i in range(1, n)])
    print(header)
    print("-" * (len(header) * 2))

    for i in range(n):
        row = f"{x_data[i]:.2f}\t"
        for j in range(n - i):
            row += f"{table[i][j]:.4f}\t"
        print(row)
    print("\n")


# --- 2. HÀM NỘI SUY GAUSS I ---
def gauss_I_loop(diff_table, n, x_0_index, y_0, t, precision):
    """
    Thực hiện nội suy Gauss I (tiến)
    Đường đi: y0 -> Δy0 -> Δ²y-1 -> Δ³y-1 -> Δ⁴y-2 ...
    """
    P_sym = y_0
    t_prod = 1
    idx = x_0_index

    print(f"Số hạng 0 (y₀): {y_0}")

    for i in range(1, n):
        fact = math.factorial(i)

        try:
            if i % 2 == 1:  # Số hạng lẻ
                t_prod = t_prod * (t - (i - 1) / 2)
                diff_val = diff_table[idx][i]
            else:  # Số hạng chẵn
                t_prod = t_prod * (t + (i - 2) / 2)
                idx -= 1
                diff_val = diff_table[idx][i]

        except IndexError:
            print(f"Dừng ở số hạng {i-1} vì hết dữ liệu sai phân.")
            break

        if diff_val == 0:
            print(f"Dừng ở số hạng {i-1} vì sai phân bằng 0.")
            break

        term = (t_prod / fact) * diff_val
        P_sym += term

        print(f"Số hạng {i}:")
        print(f"  Biểu thức t: {t_prod}")
        print(f"  Sai phân: Δ^{i}y = {diff_val:.{precision}f}")
        print(f"  Số hạng đầy đủ: ({t_prod}) / {fact} * ({diff_val:.{precision}f})")
        print(f"  P(t) hiện tại = {P_sym}")

    return P_sym

# --- 3. HÀM NỘI SUY GAUSS II ---
def gauss_II_loop(diff_table, n, x_0_index, y_0, t, precision):
    """
    Thực hiện nội suy Gauss II (lùi)
    Đường đi: y0 -> Δy-1 -> Δ²y-1 -> Δ³y-2 -> Δ⁴y-2 ...
    """
    P_sym = y_0
    t_prod = 1
    idx = x_0_index

    print(f"Số hạng 0 (y₀): {y_0}")

    for i in range(1, n):
        fact = math.factorial(i)

        try:
            if i % 2 == 1:  # Số hạng lẻ
                t_prod = t_prod * (t + (i - 1) / 2)
                idx -= 1
                diff_val = diff_table[idx][i]
            else:  # Số hạng chẵn
                t_prod = t_prod * (t - (i - 2) / 2)
                diff_val = diff_table[idx][i]

        except IndexError:
            print(f"Dừng ở số hạng {i-1} vì hết dữ liệu sai phân.")
            break

        if diff_val == 0:
            print(f"Dừng ở số hạng {i-1} vì sai phân bằng 0.")
            break

        term = (t_prod / fact) * diff_val
        P_sym += term

        print(f"Số hạng {i}:")
        print(f"  Biểu thức t: {t_prod}")
        print(f"  Sai phân: Δ^{i}y = {diff_val:.{precision}f}")
        print(f"  Số hạng đầy đủ: ({t_prod}) / {fact} * ({diff_val:.{precision}f})")
        print(f"  P(t) hiện tại = {P_sym}")

    return P_sym


# --- 4. HÀM VẼ ĐỒ THỊ ---
def plot_function(P_sym_t, t_sym, x_0, h, x_data, y_data, title):
    """Vẽ đồ thị hàm nội suy P(x) và các điểm dữ liệu gốc."""

    P_func = lambdify(t_sym, P_sym_t, 'numpy')

    x_min = min(x_data) - h
    x_max = max(x_data) + h
    x_vals_plot = np.linspace(x_min, x_max, 400)

    t_vals_plot = (x_vals_plot - x_0) / h
    y_vals_plot = P_func(t_vals_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals_plot, y_vals_plot, label='Đa thức nội suy P(x)', color='blue')
    plt.scatter(x_data, y_data, color='red', zorder=5, label='Dữ liệu gốc')
    plt.axvline(x=x_0, color='green', linestyle='--', label=f'Mốc trung tâm x₀ = {x_0}')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y = P(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 5. HÀM CHÍNH (MAIN) ---
def main():
    # --- A. NHẬP LIỆU TỪ NGƯỜI DÙNG ---

    while True:
        try:
            choice = int(input("Chọn công thức (1 cho Gauss I, 2 cho Gauss II): "))
            if choice in [1, 2]:
                break
            else:
                print("Vui lòng chỉ nhập 1 hoặc 2.")
        except ValueError:
            print("Nhập không hợp lệ, vui lòng nhập số.")

    title = "Nội suy Gauss I (Tiến)" if choice == 1 else "Nội suy Gauss II (Lùi)"
    print(f"\nBạn đã chọn: {title}")

    while True:
        try:
            n = int(input("Nhập số lượng mốc dữ liệu (n): "))
            if n > 1:
                break
            else:
                print("Cần ít nhất 2 mốc dữ liệu.")
        except ValueError:
            print("Nhập không hợp lệ, vui lòng nhập số nguyên.")

    print("\n--- NHẬP DỮ LIỆU ---")
    print("LƯU Ý: Các mốc x phải cách đều (equidistant).")
    x_data = []
    y_data = []
    h = None
    for i in range(n):
        while True:
            try:
                x = float(input(f"Nhập x[{i}]: "))
                y = float(input(f"Nhập y[{i}]: "))

                if i == 1:
                    h = x - x_data[0]
                    if h == 0:
                        print("Lỗi: h không thể bằng 0. Vui lòng nhập lại.")
                        continue
                elif i > 1:
                    if not math.isclose(x - x_data[-1], h):
                        print(f"Lỗi: Các mốc không cách đều! Khoảng cách phải là {h}.")
                        print(f"Khoảng cách hiện tại là {x - x_data[-1]}. Vui lòng nhập lại.")
                        continue

                x_data.append(x)
                y_data.append(y)
                break
            except ValueError:
                print("Nhập không hợp lệ, vui lòng nhập số.")

    while True:
        try:
            precision = int(input("\nNhập số chữ số sau dấu phẩy cho kết quả: "))
            if precision >= 0:
                break
        except ValueError:
            print("Nhập không hợp lệ.")

    # --- B. XỬ LÝ VÀ TÍNH TOÁN ---

    diff_table = create_diff_table(y_data, n)
    print_diff_table(x_data, diff_table, n)

    x_0_index = n // 2
    x_0 = x_data[x_0_index]
    y_0 = y_data[x_0_index]

    print(f"Chọn mốc trung tâm x₀ = {x_0} (tại chỉ số {x_0_index})")
    print(f"Bước h = {h}")

    t_sym = symbols('t')
    print(f"Biến t được định nghĩa là: t = (x - {x_0}) / {h}\n")

    print("--- XÂY DỰNG ĐA THỨC P(t) TỪNG BƯỚC ---")
    if choice == 1:
        P_symbolic_t = gauss_I_loop(diff_table, n, x_0_index, y_0, t_sym, precision)
    else:
        P_symbolic_t = gauss_II_loop(diff_table, n, x_0_index, y_0, t_sym, precision)

    # --- C. HIỂN THỊ KẾT QUẢ ---

    print("\n" + "="*30)
    print("--- KẾT QUẢ ĐA THỨC NỘI SUY ---")
    print(f"Đa thức theo t (trước khi rút gọn): \nP(t) = {P_symbolic_t}")

    x = symbols('x')
    t_expr = (x - x_0) / h

    P_symbolic_x = simplify(P_symbolic_t.subs(t_sym, t_expr))

    print(f"\nĐa thức theo x (sau khi rút gọn): \nP(x) = {P_symbolic_x}")
    print("="*30 + "\n")

    plot_function(P_symbolic_t, t_sym, x_0, h, x_data, y_data, title)

    # --- D. TÍNH TOÁN THÊM (ĐÃ CẢI TIẾN) ---

    # 1. Tính giá trị gần đúng P(x)
    calculate_value_at_point(P_symbolic_t, t_sym, x_0, h, precision)

    # 2. Tính đạo hàm bậc n: P^(n)(x)
    calculate_derivative_at_point(P_symbolic_t, t_sym, x_0, h, precision)

    print("\nChương trình kết thúc.")

# --- 6. HÀM CHỨC NĂNG CẢI TIẾN ---

def calculate_value_at_point(P_symbolic_t, t_sym, x_0, h, precision):
    """Hỏi người dùng và tính giá trị P(x) tại một điểm."""
    while True:
        calc_val = input("\nBạn có muốn tìm giá trị gần đúng tại một điểm x (y/n)? ").strip().lower()
        if calc_val == 'y':
            try:
                x_val = float(input("  Nhập giá trị x cần tính: "))

                # Tính t từ x
                t_val = (x_val - x_0) / h

                # Thay giá trị t_val vào đa thức P(t)
                y_val = P_symbolic_t.subs(t_sym, t_val)

                print(f"    Tại x = {x_val}")
                print(f"    Giá trị t tương ứng = ({x_val} - {x_0}) / {h} = {t_val:.{precision}f}")
                print(f"    Giá trị gần đúng P({x_val}) = {y_val:.{precision}f}")

            except ValueError:
                print("    Giá trị x không hợp lệ.")
        elif calc_val == 'n':
            break
        else:
            print("    Vui lòng nhập 'y' hoặc 'n'.")

def calculate_derivative_at_point(P_symbolic_t, t_sym, x_0, h, precision):
    """
    Hỏi người dùng bậc đạo hàm (n) và tính P^(n)(x) tại một điểm.
    Sử dụng quy tắc chuỗi: P^(n)(x) = P^(n)(t) * (1/h^n)
    """
    while True:
        calc_deriv = input("\nBạn có muốn tìm đạo hàm bậc cao tại một điểm x (y/n)? ").strip().lower()
        if calc_deriv == 'y':
            try:
                # *** YÊU CẦU CẢI TIẾN: HỎI BẬC ĐẠO HÀM ***
                deriv_order = int(input("  Nhập bậc đạo hàm (ví dụ: 1, 2, 3...): "))
                if deriv_order <= 0:
                    print("    Bậc đạo hàm phải là số nguyên dương.")
                    continue

                x_val = float(input(f"  Nhập giá trị x cần tính đạo hàm bậc {deriv_order}: "))

                # 1. Tính P^(n)(t) = d^n(P)/dt^n
                #    Sử dụng diff(biểu thức, biến, bậc)
                P_deriv_t_n = diff(P_symbolic_t, t_sym, deriv_order)

                # 2. Tính t từ x
                t_val = (x_val - x_0) / h

                # 3. Thay giá trị t_val vào P^(n)(t)
                deriv_val_t = P_deriv_t_n.subs(t_sym, t_val)

                # 4. Áp dụng quy tắc chuỗi: P^(n)(x) = P^(n)(t) * (1/h^n)
                h_n_factor = (1 / (h ** deriv_order))
                deriv_val_x = deriv_val_t * h_n_factor

                print(f"\n    --- Tính đạo hàm bậc {deriv_order} tại x = {x_val} ---")
                print(f"    Đa thức đạo hàm P^({deriv_order})(t) = {P_deriv_t_n}")
                print(f"    Quy tắc chuỗi: P^({deriv_order})(x) = P^({deriv_order})(t) * (1 / h^{deriv_order})")
                print(f"    Tại x = {x_val} (t = {t_val:.{precision}f})")
                print(f"    P^({deriv_order})(t={t_val:.{precision}f}) = {deriv_val_t:.{precision}f}")
                print(f"    P^({deriv_order})({x_val}) = {deriv_val_t:.{precision}f} * (1 / {h}^{deriv_order}) = {deriv_val_x:.{precision}f}")

            except ValueError:
                print("    Giá trị nhập không hợp lệ (bậc đạo hàm hoặc x).")
        elif calc_deriv == 'n':
            break
        else:
            print("    Vui lòng nhập 'y' hoặc 'n'.")


# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    main()