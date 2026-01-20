import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def check_monotonicity(original_x, original_y):
    """
    Kiểm tra điều kiện cần: hàm y=f(x) phải đơn điệu (có hàm ngược)
    trên đoạn đang xét .
    """
    print("\n--- 🔬 Bắt đầu kiểm tra điều kiện cần ---")

    # 1. Ghép cặp (x, y) và sắp xếp theo thứ tự x tăng dần
    try:
        data_pairs = sorted(zip(original_x, original_y))
        sorted_y = [y for x, y in data_pairs]
    except Exception as e:
        print(f"Lỗi khi sắp xếp dữ liệu: {e}")
        return False

    # 2. Kiểm tra xem y có luôn tăng (non-decreasing) không
    is_non_decreasing = all(sorted_y[i] <= sorted_y[i+1] for i in range(len(sorted_y) - 1))

    # 3. Kiểm tra xem y có luôn giảm (non-increasing) không
    is_non_increasing = all(sorted_y[i] >= sorted_y[i+1] for i in range(len(sorted_y) - 1))

    # 4. Hàm là đơn điệu nếu nó luôn tăng hoặc luôn giảm
    is_monotonic = is_non_decreasing or is_non_increasing

    if is_monotonic:
        print("✅ Đã kiểm tra: Dữ liệu (y=f(x)) là đơn điệu.")
        print("=> ĐẠT ĐIỀU KIỆN sử dụng Phương pháp Hàm ngược .")
        print("-------------------------------------------\n")
        return True
    else:
        print("❌ LỖI: Dữ liệu (y=f(x)) KHÔNG ĐƠN ĐIỆU (không luôn tăng hoặc luôn giảm).")
        print("=> KHÔNG ĐẠT ĐIỀU KIỆN. Hàm y=f(x) không có hàm ngược trên đoạn này.")
        print("   Việc tiếp tục sẽ cho kết quả xấp xỉ không chính xác.")
        print("-------------------------------------------\n")
        return False

def get_user_data():
    """
    Thu thập dữ liệu (x, y) và các tham số từ người dùng.
    Dữ liệu sẽ được hoán vị để chuẩn bị cho nội suy ngược.
    """
    print("--- 🚀 Bắt đầu chương trình Nội suy ngược (Phương pháp Hàm ngược) ---")
    print("Phương pháp này sẽ xây dựng hàm x = P(y) từ các điểm dữ liệu (y_i, x_i).\n")

    while True:
        try:
            n = int(input("1. Vui lòng nhập số lượng điểm dữ liệu (n+1): "))
            if n > 1:
                break
            print("Lỗi: Cần ít nhất 2 điểm dữ liệu.")
        except ValueError:
            print("Lỗi: Vui lòng nhập một số nguyên.")

    original_x = []
    original_y = []

    print("\n2. Vui lòng nhập các cặp điểm dữ liệu (x, y):")
    for i in range(n):
        while True:
            try:
                x_val = float(input(f"   Nhập x[{i}]: "))
                y_val = float(input(f"   Nhập y[{i}]: "))
                original_x.append(x_val)
                original_y.append(y_val)
                break
            except ValueError:
                print("Lỗi: Vui lòng nhập giá trị số hợp lệ.")

    while True:
        try:
            precision = int(input("\n3. Nhập số chữ số sau dấu phẩy (sai số) bạn muốn hiển thị: "))
            if precision >= 0:
                break
            print("Lỗi: Vui lòng nhập một số không âm.")
        except ValueError:
            print("Lỗi: Vui lòng nhập một số nguyên.")

    # yêu cầu sử dụng các mốc (y_i, x_i)
    # Vì vậy, đối với hàm nội suy của chúng ta:
    # 'y_points_for_inverse' (để nội suy) là 'original_y'
    # 'x_points_for_inverse' (để nội suy) là 'original_x'
    y_points_for_inverse = np.array(original_y)
    x_points_for_inverse = np.array(original_x)

    return y_points_for_inverse, x_points_for_inverse, precision, np.array(original_x), np.array(original_y)


def build_inverse_polynomial(y_points, x_points):
    """
    Xây dựng đa thức nội suy Lagrange tượng trưng x = P(y)
    sử dụng SymPy.
    """
    y = sp.symbols('y')
    P_y = 0

    n = len(y_points)
    lagrange_terms = []

    print("--- 📖 Hiển thị các bước giải toán học ---")
    print(f"Sử dụng Phương pháp Hàm ngược , ta xấp xỉ x = f_inv(y) .")
    print(f"Ta xây dựng đa thức nội suy P(y) từ các cặp điểm đã hoán vị (y_i, x_i) .")
    print("Dữ liệu dùng để nội suy: ")
    for i in range(n):
        print(f"   (y_{i}={y_points[i]}, x_{i}={x_points[i]})")

    print("\nĐa thức nội suy Lagrange có dạng: x = P(y) = Σ [x_i * L_i(y)]")

    for i in range(n):
        L_i = 1
        numerator = 1
        denominator = 1

        for j in range(n):
            if i != j:
                numerator *= (y - y_points[j])
                denominator *= (y_points[i] - y_points[j])

        L_i = numerator / denominator
        lagrange_terms.append(L_i)

        print(f"\nTerm L_{i}(y) cho x_{i}={x_points[i]}:")
        print(f"   L_{i}(y) = {sp.expand(L_i)}") # In L_i(y) đã rút gọn

        P_y += x_points[i] * L_i

    # Rút gọn đa thức cuối cùng
    P_y_expanded = sp.expand(P_y)

    print("\n--------------------------------------------------")
    print("✅ HÀM SỐ NỘI SUY NGƯỢC (Đa thức cuối cùng):")
    print(f"   x = P(y) = {P_y_expanded}")
    print("--------------------------------------------------\n")

    return P_y_expanded, y

def plot_inverse_interpolation(P_y_symbolic, y_symbol, original_x, original_y):
    """
    Vẽ đồ thị hàm số x = P(y) và các điểm dữ liệu gốc (x, y).
    """
    print("Đang tạo đồ thị...")

    # Chuyển hàm SymPy thành hàm số có thể tính toán bằng Numpy
    P_y_numeric = sp.lambdify(y_symbol, P_y_symbolic, 'numpy')

    # Tạo một dải giá trị y để vẽ đồ thị
    # Sử dụng y_points gốc (original_y) để xác định phạm vi vẽ
    y_plot_values = np.linspace(min(original_y), max(original_y), 400)

    # Tính các giá trị x tương ứng
    x_plot_values = P_y_numeric(y_plot_values)

    plt.figure(figsize=(10, 6))
    # Vẽ hàm nội suy
    plt.plot(x_plot_values, y_plot_values, label=f'Hàm nội suy x = P(y)', color='blue')
    # Vẽ các điểm dữ liệu gốc (x_i, y_i)
    plt.scatter(original_x, original_y, color='red', zorder=5, label='Các điểm dữ liệu gốc (x_i, y_i)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Đồ thị Nội suy ngược (Phương pháp Hàm ngược)')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()

def calculate_x_from_y(P_y_symbolic, y_symbol, precision, y_points):
    """
    Hỏi người dùng giá trị y và tính giá trị x tương ứng.
    """
    while True:
        choice = input("Bạn có muốn tìm giá trị x khi biết trước một giá trị y không? (c/k): ").strip().lower()
        if choice == 'c':
            try:
                # Gợi ý một giá trị y trung bình từ dữ liệu gốc
                y_bar = float(input(f"   Nhập giá trị y (ví dụ: y = {round(np.mean(y_points), 2)}): "))

                # Tính toán giá trị
                x_result = P_y_symbolic.subs(y_symbol, y_bar)

                print(f"   Khi y = {y_bar}:")
                print(f"   x = P({y_bar})")
                print(f"   x ≈ {x_result.evalf(precision)}")
                break
            except ValueError:
                print("Lỗi: Vui lòng nhập một giá trị số.")
        elif choice == 'k':
            break
        else:
            print("Lỗi: Vui lòng chỉ nhập 'c' (có) hoặc 'k' (không).")

def calculate_derivative(P_y_symbolic, y_symbol, precision):
    """
    Hỏi người dùng về bậc đạo hàm (m), tính toán và hiển thị nó.
    Sau đó, hỏi giá trị y để tính giá trị của đạo hàm.
    """
    while True:
        choice = input("\nBạn có muốn tính đạo hàm (theo y) của hàm x = P(y) không? (c/k): ").strip().lower()
        if choice == 'c':
            try:
                m = int(input("   Nhập bậc đạo hàm m (ví dụ: 1, 2, ...): "))
                if m < 0:
                    print("Lỗi: Bậc đạo hàm phải là số không âm.")
                    continue

                # Tính đạo hàm bậc m
                P_deriv = sp.diff(P_y_symbolic, y_symbol, m)

                print(f"\n   Hàm đạo hàm bậc {m} (d^m(x) / dy^{m}):")
                print(f"   P'({y_symbol.name}) = {P_deriv}")

                # Hỏi để tính giá trị
                while True:
                    val_choice = input(f"\n   Bạn có muốn tính giá trị của đạo hàm bậc {m} tại một điểm y không? (c/k): ").strip().lower()
                    if val_choice == 'c':
                        try:
                            y_val = float(input(f"      Nhập giá trị y để tính đạo hàm: "))
                            deriv_result = P_deriv.subs(y_symbol, y_val)
                            print(f"      Giá trị đạo hàm tại y = {y_val} là:")
                            print(f"      P'({y_val}) ≈ {deriv_result.evalf(precision)}")
                            break
                        except ValueError:
                            print("      Lỗi: Vui lòng nhập một giá trị số.")
                    elif val_choice == 'k':
                        break
                    else:
                        print("      Lỗi: Vui lòng chỉ nhập 'c' hoặc 'k'.")
                break
            except ValueError:
                print("Lỗi: Vui lòng nhập một số nguyên cho bậc đạo hàm.")
        elif choice == 'k':
            break
        else:
            print("Lỗi: Vui lòng chỉ nhập 'c' (có) hoặc 'k' (không).")


def main():
    """
    Hàm chính điều khiển luồng của chương trình.
    """
    try:
        # 1. Thu thập dữ liệu
        # y_points_inv, x_points_inv là dữ liệu đã hoán vị (y_i, x_i) để nội suy
        # original_x, original_y là dữ liệu gốc (x_i, y_i) để kiểm tra và vẽ đồ thị
        y_points_inv, x_points_inv, precision, original_x, original_y = get_user_data()

        # 2. *** KIỂM TRA ĐIỀU KIỆN CẦN ***
        if not check_monotonicity(original_x, original_y):
            print("Chương trình dừng lại do không thỏa mãn điều kiện.")
            return # Dừng chương trình

        # 3. Xây dựng đa thức và hiển thị các bước
        # Sử dụng dữ liệu đã hoán vị để xây dựng P(y)
        P_y, y_sym = build_inverse_polynomial(y_points_inv, x_points_inv)

        # 4. Vẽ đồ thị
        # Sử dụng dữ liệu gốc để vẽ các điểm (x_i, y_i)
        plot_inverse_interpolation(P_y, y_sym, original_x, original_y)

        # 5. Tính x khi biết y
        # Truyền y_points_inv (là original_y) để gợi ý giá trị trung bình
        calculate_x_from_y(P_y, y_sym, precision, y_points_inv)

        # 6. Tính đạo hàm
        calculate_derivative(P_y, y_sym, precision)

        print("\n--- 👋 Kết thúc chương trình ---")

    except Exception as e:
        print(f"\nĐã xảy ra lỗi không mong muốn: {e}")
    except KeyboardInterrupt:
        print("\n\nChương trình đã bị ngắt bởi người dùng. Tạm biệt!")


if __name__ == "__main__":
    main()