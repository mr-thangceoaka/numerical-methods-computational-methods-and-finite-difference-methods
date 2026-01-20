import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from math import factorial
import warnings

# Bỏ qua các cảnh báo RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_input_data():
    """Hàm nhập dữ liệu từ người dùng."""
    print("--- NHẬP DỮ LIỆU ---")
    while True:
        try:
            n = int(input("Nhập số lượng mốc nội suy (n+1 điểm): "))
            if n > 1:
                break
            print("Cần ít nhất 2 điểm dữ liệu.")
        except ValueError:
            print("Vui lòng nhập một số nguyên.")

    x_data = np.zeros(n)
    y_data = np.zeros(n)
    print("Nhập các cặp giá trị (x, y):")
    for i in range(n):
        while True:
            try:
                x_val = float(input(f"  x[{i}]: "))
                y_val = float(input(f"  y[{i}]: "))
                x_data[i] = x_val
                y_data[i] = y_val
                break
            except ValueError:
                print("Lỗi: Vui lòng nhập số hợp lệ.")
    return x_data, y_data, n

def check_equidistant(x_nodes):
    """
    KIỂM TRA ĐIỀU KIỆN 1: MỐC CÁCH ĐỀU
    Trả về bước nhảy h nếu cách đều, ngược lại trả về None.
    """
    print("\n--- Kiểm tra Điều kiện 1: Mốc cách đều ---")
    if len(x_nodes) < 2:
        return None

    # Sử dụng np.isclose để xử lý sai số dấu phẩy động
    h = x_nodes[1] - x_nodes[0]
    for i in range(1, len(x_nodes) - 1):
        if not np.isclose(x_nodes[i+1] - x_nodes[i], h):
            print(f"LỖI: Mốc không cách đều.")
            print(f"Khoảng cách (x[1]-x[0]) = {h}")
            print(f"Khoảng cách (x[{i+1}]-x[{i}]) = {x_nodes[i+1] - x_nodes[i]} (Khác biệt)")
            return None

    print(f"HỢP LỆ: Các mốc x cách đều với bước nhảy h = {h}.")
    return h

def create_diff_tables(y_data, n):
    """Tạo bảng sai phân hữu hạn tiến và lùi."""
    f_diff = np.zeros((n, n))
    f_diff[:, 0] = y_data
    for k in range(1, n):
        for i in range(n - k):
            f_diff[i, k] = f_diff[i+1, k-1] - f_diff[i, k-1]

    b_diff = np.zeros((n, n))
    b_diff[:, 0] = y_data
    for k in range(1, n):
        for i in range(k, n):
            b_diff[i, k] = b_diff[i, k-1] - b_diff[i-1, k-1]

    return f_diff, b_diff

def build_symbolic_poly(x_data, y_data, h, f_diff_table, n):
    """
    Xây dựng đa thức nội suy Newton tiến dạng tượng trưng (symbolic).
    """
    x = sp.symbols('x')
    x0 = x_data[0]
    t = (x - x0) / h

    poly = y_data[0]
    term = 1

    for k in range(1, n):
        term *= (t - (k - 1))
        poly_term = (f_diff_table[0, k] / factorial(k)) * term
        poly += poly_term

    poly_simplified = sp.expand(poly)
    return poly_simplified, x

def plot_polynomial(poly, x_symbol, x_data, y_data):
    """Vẽ đồ thị đa thức nội suy và các điểm dữ liệu."""
    print("--- 3. Đồ thị hàm số ---")
    poly_func = sp.lambdify(x_symbol, poly, 'numpy')

    x_min = np.min(x_data)
    x_max = np.max(x_data)
    x_range = np.linspace(x_min - 0.5, x_max + 0.5, 400)
    y_range = poly_func(x_range)

    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_range, label=f"Đa thức nội suy P(x)")
    plt.scatter(x_data, y_data, color='red', zorder=5, label="Các điểm dữ liệu gốc")
    plt.xlabel("x")
    plt.ylabel("y = P(x)")
    plt.title("Đồ thị Đa thức Nội suy Newton")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()
    print("Đã hiển thị đồ thị.")

def handle_inverse_interpolation(x_data, y_data, h, f_diff, b_diff, n):
    """
    Thực hiện nội suy ngược bằng phương pháp lặp.
    Hàm này sẽ hỏi người dùng giá trị y_bar, sai số, và phương pháp (tiến/lùi).
    """
    print("\n--- 2. Nội suy ngược (Phương pháp lặp) ---")
    try:
        y_bar = float(input("Nhập giá trị y (y_bar) cần tìm x: "))
        decimals = int(input("Nhập số chữ số chính xác sau dấu phẩy (cho sai số): "))
        epsilon = 10**(-decimals)
    except ValueError:
        print("Lỗi: Vui lòng nhập số hợp lệ.")
        return

    # --- KIỂM TRA ĐIỀU KIỆN 2: KHOẢNG CÁCH LY ---
    print(f"\n--- Kiểm tra Điều kiện 2: Khoảng cách ly nghiệm cho y_bar = {y_bar} ---")

    # MỚI: Tìm TẤT CẢ các khoảng cách ly
    found_intervals = []
    for i in range(n - 1):
        y_i = y_data[i]
        y_i_plus_1 = y_data[i+1]

        # Kiểm tra xem y_bar có nằm giữa y_i và y_i+1 (kể cả khi tăng hoặc giảm)
        if (y_i <= y_bar <= y_i_plus_1) or (y_i >= y_bar >= y_i_plus_1):
            # Kiểm tra để đảm bảo khoảng không bị "dẹt" (y_i == y_i+1)
            # nếu y_bar cũng bằng y_i, có thể gây chia cho 0 ở Delta(y)
            if not np.isclose(y_i, y_i_plus_1):
                found_intervals.append(i) # Lưu chỉ số k

    k = -1 # Chỉ số của mốc bắt đầu (x_k)

    if len(found_intervals) == 0:
        print(f"LỖI: Không tìm thấy khoảng cách ly nghiệm nào chứa y_bar = {y_bar}.")
        print("Giá trị y nằm ngoài phạm vi dữ liệu hoặc rơi vào khoảng không đơn điệu.")
        return

    elif len(found_intervals) == 1:
        k = found_intervals[0]
        print(f"HỢP LỆ: Tìm thấy 1 khoảng cách ly nghiệm.")

    else: # > 1 khoảng
        print(f"CẢNH BÁO: Tìm thấy {len(found_intervals)} khoảng cách ly nghiệm (do hàm không đơn điệu).")
        print("Vui lòng chọn 1 khoảng để tiến hành nội suy:")
        for idx, k_val in enumerate(found_intervals):
            print(f"  {idx+1}. Khoảng (x_{k_val}, x_{k_val+1}) = ({x_data[k_val]}, {x_data[k_val+1]})")

        while True:
            try:
                choice_idx = int(input(f"Chọn số (1-{len(found_intervals)}): ")) - 1
                if 0 <= choice_idx < len(found_intervals):
                    k = found_intervals[choice_idx]
                    break
                else:
                    print("Lựa chọn không hợp lệ.")
            except ValueError:
                print("Vui lòng nhập một số.")

    print(f"Đã chọn khoảng (x_{k}, x_{k+1}) = ({x_data[k]}, {x_data[k+1]}) để tính toán.")
    # --- KẾT THÚC KIỂM TRA KHOẢNG CÁCH LY ---


    choice = input("Sử dụng lặp Newton 'tiến' (xuất phát từ x_k) hay 'lùi' (xuất phát từ x_k+1)? (gõ 'tien' hoặc 'lui'): ").strip().lower()

    t = sp.symbols('t')
    iterations = []
    max_iter = 100
    t_current = 0.0
    x_result = 0.0

    if choice == 'tien':
        print(f"Sử dụng công thức lặp Newton tiến, xuất phát từ x_k = {x_data[k]}.")

        # Kiểm tra điều kiện chia cho 0
        if np.isclose(f_diff[k, 1], 0):
            print(f"LỖI: Không thể lặp tiến. Sai phân Delta(y_k) = {f_diff[k, 1]} quá gần 0.")
            return

        phi_t_terms = 0
        for i in range(2, n - k):
            term = 1
            for j in range(i):
                term *= (t - j)
            phi_t_terms += (f_diff[k, i] / factorial(i)) * term

        phi_t = (y_bar - y_data[k]) / f_diff[k, 1] - (1 / f_diff[k, 1]) * phi_t_terms
        phi_func = sp.lambdify(t, phi_t, 'numpy')

        t_current = (y_bar - y_data[k]) / f_diff[k, 1]

        print("\n--- Các bước lặp (Tiến) ---")
        print(f"Hàm lặp: t_j+1 = phi(t_j) = {phi_t}")
        print(f"Giá trị ban đầu: t_0 = (y_bar - y_k) / Delta(y_k) = ({y_bar} - {y_data[k]}) / {f_diff[k, 1]:.5f} = {t_current:.{decimals+4}f}")

        for j in range(max_iter):
            t_next = phi_func(t_current)
            iterations.append((j, t_current, t_next))
            if abs(t_next - t_current) < epsilon:
                t_current = t_next
                break
            t_current = t_next

        x_result = x_data[k] + t_current * h

    elif choice == 'lui':
        print(f"Sử dụng công thức lặp Newton lùi, xuất phát từ x_k+1 = {x_data[k+1]}.")

        # Kiểm tra điều kiện chia cho 0
        if np.isclose(b_diff[k+1, 1], 0):
            print(f"LỖI: Không thể lặp lùi. Sai phân Nabla(y_k+1) = {b_diff[k+1, 1]} quá gần 0.")
            return

        psi_t_terms = 0
        for i in range(2, k + 2):
            term = 1
            for j in range(i):
                term *= (t + j)
            psi_t_terms += (b_diff[k+1, i] / factorial(i)) * term

        psi_t = (y_bar - y_data[k+1]) / b_diff[k+1, 1] - (1 / b_diff[k+1, 1]) * psi_t_terms
        psi_func = sp.lambdify(t, psi_t, 'numpy')

        t_current = (y_bar - y_data[k+1]) / b_diff[k+1, 1]

        print("\n--- Các bước lặp (Lùi) ---")
        print(f"Hàm lặp: t_j+1 = psi(t_j) = {psi_t}")
        print(f"Giá trị ban đầu: t_0 = (y_bar - y_k+1) / Nabla(y_k+1) = ({y_bar} - {y_data[k+1]}) / {b_diff[k+1, 1]:.5f} = {t_current:.{decimals+4}f}")

        for j in range(max_iter):
            t_next = psi_func(t_current)
            iterations.append((j, t_current, t_next))
            if abs(t_next - t_current) < epsilon:
                t_current = t_next
                break
            t_current = t_next

        x_result = x_data[k+1] + t_current * h

    else:
        print("Lựa chọn không hợp lệ. Vui lòng gõ 'tien' hoặc 'lui'.")
        return

    # --- Hiển thị kết quả lặp ---
    num_iter = len(iterations)
    print("\n--- Quá trình lặp ---")
    if num_iter > 20:
        print(f"Tổng cộng {num_iter} lần lặp. Hiển thị 10 lần đầu và 10 lần cuối:")
        for i, t_j, t_j1 in iterations[:10]:
            print(f"  Lần {i}: t_{i} = {t_j:.{decimals+4}f} -> t_{i+1} = {t_j1:.{decimals+4}f}")
        print("  ...")
        for i, t_j, t_j1 in iterations[-10:]:
            print(f"  Lần {i}: t_{i} = {t_j:.{decimals+4}f} -> t_{i+1} = {t_j1:.{decimals+4}f}")
    else:
        for i, t_j, t_j1 in iterations:
            print(f"  Lần {i}: t_{i} = {t_j:.{decimals+4}f} -> t_{i+1} = {t_j1:.{decimals+4}f}")

    print("\n--- Kết quả Nội suy ngược ---")
    if num_iter == max_iter:
        print(f"CẢNH BÁO: Lặp đạt tối đa {max_iter} lần nhưng chưa hội tụ!")

    print(f"Sau {num_iter} lần lặp, giá trị t hội tụ: t = {t_current:.{decimals+4}f}")

    if choice == 'tien':
        print(f"Giá trị x_bar = x_k + t*h = {x_data[k]} + ({t_current:.{decimals+4}f}) * {h}")
    else: # lùi
        print(f"Giá trị x_bar = x_k+1 + t*h = {x_data[k+1]} + ({t_current:.{decimals+4}f}) * {h}")

    print(f"==> Vậy, với y = {y_bar}, ta tìm được x ≈ {x_result:.{decimals}f}")

def handle_forward_evaluation(poly, x_symbol):
    """Tính P(x) tại một giá trị x do người dùng nhập."""
    print("\n--- 4. Tính giá trị hàm số (Nội suy xuôi) ---")
    while True:
        try:
            x_val_str = input("Bạn có muốn tính giá trị P(x) tại một điểm x cụ thể không? (nhập giá trị x hoặc bỏ trống để bỏ qua): ")
            if not x_val_str:
                return
            x_val = float(x_val_str)
            poly_func = sp.lambdify(x_symbol, poly, 'numpy')
            y_val = poly_func(x_val)
            print(f"Kết quả: P({x_val}) = {y_val:.7f}")
            break
        except ValueError:
            print("Lỗi: Vui lòng nhập một số hợp lệ.")
        except Exception as e:
            print(f"Lỗi khi tính toán: {e}")

def handle_derivative(poly, x_symbol):
    """Tính đạo hàm cấp m và giá trị của nó."""
    print("\n--- 5. Tính đạo hàm ---")
    while True:
        try:
            m_str = input("Bạn có muốn tính đạo hàm không? (nhập cấp đạo hàm m, ví dụ: 1, 2... hoặc bỏ trống để bỏ qua): ")
            if not m_str:
                return
            m = int(m_str)
            if m < 0:
                print("Cấp đạo hàm phải là số không âm.")
                continue

            deriv = poly
            for _ in range(m):
                deriv = sp.diff(deriv, x_symbol)

            deriv_simplified = sp.expand(deriv)
            print(f"Đạo hàm cấp {m}, P^({m})(x) = {deriv_simplified}")

            x_val_str = input(f"Nhập giá trị x để tính P^({m})(x) (hoặc bỏ trống để bỏ qua): ")
            if not x_val_str:
                break

            x_val = float(x_val_str)
            deriv_func = sp.lambdify(x_symbol, deriv_simplified, 'numpy')
            deriv_val = deriv_func(x_val)
            print(f"Kết quả: P^({m})({x_val}) = {deriv_val:.7f}")
            break

        except ValueError:
            print("Lỗi: Vui lòng nhập một số nguyên hợp lệ.")
        except Exception as e:
            print(f"Lỗi khi tính toán: {e}")

def main():
    """Hàm điều khiển chính của chương trình."""

    # 1. Nhập dữ liệu
    x_data, y_data, n = get_input_data()
    print("\nDữ liệu đã nhập:")
    print("X:", x_data)
    print("Y:", y_data)

    # 2. KIỂM TRA ĐIỀU KIỆN 1: MỐC CÁCH ĐỀU
    h = check_equidistant(x_data)
    if h is None:
        print("\nLỖI: Điều kiện tiên quyết 'Mốc cách đều' không được đáp ứng.")
        print("Không thể sử dụng Phương pháp lặp Newton. Vui lòng chạy lại.")
        return

    # 3. Tạo bảng sai phân
    f_diff, b_diff = create_diff_tables(y_data, n)

    # 4. Xây dựng, in và vẽ đồ thị đa thức
    try:
        poly_symbolic, x_sym = build_symbolic_poly(x_data, y_data, h, f_diff, n)
        print("\n--- 1. Đa thức nội suy P(x) ---")
        print(f"Đa thức nội suy P(x) = {poly_symbolic}")
        plot_polynomial(poly_symbolic, x_sym, x_data, y_data)
    except Exception as e:
        print(f"Lỗi khi xây dựng hoặc vẽ đồ thị đa thức: {e}")
        return

    # 5. Thực hiện Nội suy ngược (Chức năng chính VỚI KIỂM TRA ĐIỀU KIỆN 2)
    handle_inverse_interpolation(x_data, y_data, h, f_diff, b_diff, n)

    # 6. Thực hiện Nội suy xuôi (Tính P(x))
    handle_forward_evaluation(poly_symbolic, x_sym)

    # 7. Tính đạo hàm
    handle_derivative(poly_symbolic, x_sym)

    print("\n--- Chương trình kết thúc ---")

if __name__ == "__main__":
    main()