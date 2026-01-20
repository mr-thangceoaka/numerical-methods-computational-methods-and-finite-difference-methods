import numpy as np
from scipy.integrate import quad
import sympy as sp

def clear_screen():
    print("\n" + "="*80 + "\n")

# ============================================================================
# PHẦN XỬ LÝ HÀM SỐ - ĐƠN GIẢN HÓA
# ============================================================================

def convert_to_python(expr_str):
    """
    Chuyển đổi biểu thức toán học đơn giản sang Python
    - Dùng ^ cho lũy thừa (không cần **)
    - Dùng e cho số Euler
    - ln(x) = logarit tự nhiên
    - log_10(x) = logarit cơ số 10
    - log_2(x) = logarit cơ số 2
    - Tự động thêm * giữa số và biến (2x -> 2*x)
    """
    import re

    expr_str = expr_str.strip()

    # Kiểm tra ngoặc đóng/mở
    open_count = expr_str.count('(')
    close_count = expr_str.count(')')
    if open_count != close_count:
        raise ValueError(f"Lỗi cú pháp: Thiếu ngoặc đóng ')' hoặc ngoặc mở '('\n"
                         f"Số ngoặc mở: {open_count}, Số ngoặc đóng: {close_count}\n"
                         f"Biểu thức: {expr_str}")

    # Nếu đã là lambda, return luôn
    if expr_str.startswith('lambda'):
        expr_str = expr_str.replace('^', '**')
        return eval(expr_str)

    # BƯỚC 1: Xử lý log_<cơ số> TRƯỚC tất cả (để không bị conflict với ln, log)
    # log_10(...) -> __LOG10__, log_2(...) -> __LOG2__
    expr_str = expr_str.replace('log_10', '__LOG10__')
    expr_str = expr_str.replace('log_2', '__LOG2__')
    expr_str = expr_str.replace('log_e', '__LOGE__')  # log_e = ln

    # Xử lý log_<n> tổng quát (n khác 2, 10, e)
    # log_3(x), log_5(x), etc. -> np.log(x)/np.log(n)
    # Nhưng để đơn giản, chỉ hỗ trợ log_2, log_10, log_e

    # BƯỚC 2: Tự động thêm * giữa số và (biến/hàm/ngoặc)
    # 2x -> 2*x, 3sin -> 3*sin, 2( -> 2*(
    expr_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr_str)
    # Cũng thêm * giữa ) và số/biến: )x -> )*x, )2 -> )*2
    expr_str = re.sub(r'\)(\d)', r')*\1', expr_str)
    expr_str = re.sub(r'\)([a-zA-Z])', r')*\1', expr_str)

    # BƯỚC 3: Thay thế hàm toán học cơ bản
    # Dùng placeholder để tránh xung đột
    expr_str = expr_str.replace('ln', '__LN__')
    expr_str = expr_str.replace('sin', '__SIN__')
    expr_str = expr_str.replace('cos', '__COS__')
    expr_str = expr_str.replace('tan', '__TAN__')

    # BƯỚC 4: Thay ^ thành **
    expr_str = expr_str.replace('^', '**')

    # BƯỚC 5: Xử lý số e (Euler)
    expr_str = re.sub(r'\be\b', '__E__', expr_str)

    # BƯỚC 6: Thay placeholder thành numpy functions
    replacements = {
        '__LN__': 'np.log',
        '__LOG10__': 'np.log10',
        '__LOG2__': 'np.log2',
        '__LOGE__': 'np.log',  # log_e = ln
        '__SIN__': 'np.sin',
        '__COS__': 'np.cos',
        '__TAN__': 'np.tan',
        '__E__': 'np.e',
    }

    for placeholder, func in replacements.items():
        expr_str = expr_str.replace(placeholder, func)

    # Tạo lambda function
    try:
        f = eval(f'lambda x: {expr_str}')
        # Test để đảm bảo hàm hoạt động
        try:
            test_val = f(2.0)
            if not np.isfinite(test_val):
                raise ValueError("Hàm cho giá trị không xác định (inf/nan)")
        except ZeroDivisionError:
            raise ValueError("Lỗi: Chia cho 0. Kiểm tra lại biểu thức.")
        except (ValueError, RuntimeWarning) as ve:
            if "math domain error" in str(ve) or "invalid value" in str(ve):
                raise ValueError("Lỗi: Giá trị không xác định (log số âm/0, sqrt số âm). "
                                 "Hàm có thể chỉ hoạt động trên một miền xác định cụ thể.")
            raise
        return f
    except SyntaxError as e:
        raise ValueError(f"Lỗi cú pháp: {e}\nChuỗi đã chuyển: lambda x: {expr_str}\n"
                         f"Gợi ý: Kiểm tra dấu ngoặc, phép toán, tên hàm")
    except Exception as e:
        raise ValueError(f"Lỗi: {e}\nChuỗi đã chuyển: lambda x: {expr_str}")

def convert_to_sympy(expr_str):
    """Chuyển biểu thức thành SymPy để tính đạo hàm"""
    import re

    expr_str = expr_str.strip()

    # Bỏ lambda nếu có
    if expr_str.startswith('lambda x:'):
        expr_str = expr_str[9:].strip()

    # Tự động thêm * giữa số và (biến/hàm/ngoặc)
    expr_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr_str)
    expr_str = re.sub(r'\)(\d)', r')*\1', expr_str)
    expr_str = re.sub(r'\)([a-zA-Z])', r')*\1', expr_str)

    # Xử lý log_<cơ số>
    # log_10(x) -> log(x, 10), log_2(x) -> log(x, 2)
    expr_str = re.sub(r'log_(\d+)\(([^)]+)\)', r'log(\2, \1)', expr_str)

    # Thay thế hàm: ln -> log (SymPy dùng log cho ln)
    expr_str = expr_str.replace('ln(', 'log(')

    # Thay thế ^ thành **
    expr_str = expr_str.replace('^', '**')

    # Xử lý e - thay thành E (SymPy dùng E cho số Euler)
    expr_str = re.sub(r'\be\b', 'E', expr_str)

    # Thay thế np. nếu có
    expr_str = expr_str.replace('np.', '')

    try:
        x = sp.Symbol('x')
        return sp.sympify(expr_str)
    except:
        try:
            return sp.parse_expr(expr_str, transformations='all')
        except Exception as e:
            raise ValueError(f"Không thể chuyển sang SymPy: {e}\nBiểu thức: {expr_str}")

# ============================================================================
# PHẦN TÍNH ĐẠO HÀM
# ============================================================================

def dao_ham_2_diem_can_trai(y_k, y_k1, h):
    return (y_k1 - y_k) / h

def dao_ham_2_diem_can_phai(y_k, y_k_1, h):
    return (y_k - y_k_1) / h

def dao_ham_3_diem_can_trai(y_k, y_k1, y_k2, h):
    return (-3*y_k + 4*y_k1 - y_k2) / (2*h)

def dao_ham_3_diem_trung_tam(y_k_1, y_k1, h):
    return (y_k1 - y_k_1) / (2*h)

def dao_ham_3_diem_can_phai(y_k_2, y_k_1, y_k, h):
    return (y_k_2 - 4*y_k_1 + 3*y_k) / (2*h)

def nhap_du_lieu_dao_ham():
    print("\n--- NHẬP DỮ LIỆU ---")
    print("1. Nhập tay dữ liệu (x, y)")
    print("2. Đọc từ file")

    choice = input("\nChọn (1/2): ").strip()

    if choice == "1":
        n = int(input("Số điểm dữ liệu: "))
        x_data = []
        y_data = []
        print("Nhập dữ liệu:")
        for i in range(n):
            x = float(input(f"  x[{i}] = "))
            y = float(input(f"  y[{i}] = "))
            x_data.append(x)
            y_data.append(y)
        return np.array(x_data), np.array(y_data)
    else:
        filepath = input("Đường dẫn file: ").strip()
        try:
            data = np.loadtxt(filepath)
            if data.ndim == 1:
                raise ValueError("File phải có 2 cột (x và y)")
            return data[:, 0], data[:, 1]
        except Exception as e:
            print(f"Lỗi: {e}")
            return nhap_du_lieu_dao_ham()

def tinh_dao_ham():
    clear_screen()
    print("=== TÍNH GẦN ĐÚNG ĐẠO HÀM ===\n")

    x_data, y_data = nhap_du_lieu_dao_ham()

    print("\n--- DỮ LIỆU ---")
    print(f"{'i':>3} {'x':>10} {'y':>10}")
    print("-" * 25)
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        print(f"{i:>3} {x:>10.4f} {y:>10.4f}")

    if len(x_data) > 1:
        h = x_data[1] - x_data[0]
        print(f"\nBước h = {h:.6f}")
    else:
        print("Cần ít nhất 2 điểm!")
        return

    print("\n--- CHỌN PHƯƠNG PHÁP ---")
    print("1. Công thức 2 điểm")
    print("2. Công thức 3 điểm")
    print("3. Tính tại tất cả các điểm")

    method = input("\nChọn (1/2/3): ").strip()

    if method == "1":
        print("\n--- CÔNG THỨC 2 ĐIỂM ---")
        print("a. Cận trái: y'(x_k) = (y_{k+1} - y_k) / h")
        print("b. Cận phải: y'(x_k) = (y_k - y_{k-1}) / h")
        sub = input("Chọn (a/b): ").strip().lower()
        k = int(input("Chỉ số k: "))

        if sub == "a":
            if k >= len(x_data) - 1:
                print("Không thể dùng cận trái cho điểm cuối!")
                return
            result = dao_ham_2_diem_can_trai(y_data[k], y_data[k+1], h)
            print(f"\ny'({x_data[k]:.4f}) ≈ {result:.6f}")
        else:
            if k == 0:
                print("Không thể dùng cận phải cho điểm đầu!")
                return
            result = dao_ham_2_diem_can_phai(y_data[k], y_data[k-1], h)
            print(f"\ny'({x_data[k]:.4f}) ≈ {result:.6f}")

    elif method == "2":
        print("\n--- CÔNG THỨC 3 ĐIỂM ---")
        print("a. Cận trái: y'(x_k) = (-3y_k + 4y_{k+1} - y_{k+2}) / (2h)")
        print("b. Trung tâm: y'(x_k) = (y_{k+1} - y_{k-1}) / (2h)")
        print("c. Cận phải: y'(x_k) = (y_{k-2} - 4y_{k-1} + 3y_k) / (2h)")
        sub = input("Chọn (a/b/c): ").strip().lower()
        k = int(input("Chỉ số k: "))

        if sub == "a":
            if k >= len(x_data) - 2:
                print("Không đủ điểm!")
                return
            result = dao_ham_3_diem_can_trai(y_data[k], y_data[k+1], y_data[k+2], h)
            print(f"\ny'({x_data[k]:.4f}) ≈ {result:.6f}")
        elif sub == "b":
            if k == 0 or k >= len(x_data) - 1:
                print("Không đủ điểm!")
                return
            result = dao_ham_3_diem_trung_tam(y_data[k-1], y_data[k+1], h)
            print(f"\ny'({x_data[k]:.4f}) ≈ {result:.6f}")
        else:
            if k < 2:
                print("Không đủ điểm!")
                return
            result = dao_ham_3_diem_can_phai(y_data[k-2], y_data[k-1], y_data[k], h)
            print(f"\ny'({x_data[k]:.4f}) ≈ {result:.6f}")

    else:
        print("\n" + "="*70)
        print("KẾT QUẢ ĐẠO HÀM TẠI CÁC ĐIỂM")
        print("="*70)
        print(f"{'k':>3} {'x_k':>10} {'y\'(x_k)':>12} {'Phương pháp':>30}")
        print("-" * 70)

        for k in range(len(x_data)):
            if k == 0:
                if len(x_data) >= 3:
                    result = dao_ham_3_diem_can_trai(y_data[0], y_data[1], y_data[2], h)
                    method_name = "3 điểm cận trái"
                else:
                    result = dao_ham_2_diem_can_trai(y_data[0], y_data[1], h)
                    method_name = "2 điểm cận trái"
            elif k == len(x_data) - 1:
                if len(x_data) >= 3:
                    result = dao_ham_3_diem_can_phai(y_data[k-2], y_data[k-1], y_data[k], h)
                    method_name = "3 điểm cận phải"
                else:
                    result = dao_ham_2_diem_can_phai(y_data[k], y_data[k-1], h)
                    method_name = "2 điểm cận phải"
            else:
                result = dao_ham_3_diem_trung_tam(y_data[k-1], y_data[k+1], h)
                method_name = "3 điểm trung tâm"

            print(f"{k:>3} {x_data[k]:>10.4f} {result:>12.6f} {method_name:>30}")

# ============================================================================
# PHẦN TÍCH PHÂN - VỚI TÍNH M2, M4 TỰ ĐỘNG
# ============================================================================

def tinh_M(f_expr_sympy, order, a, b, num_points=1000):
    """
    Tính M_n = max|f^(n)(x)| trên [a,b]
    Hiển thị công thức đạo hàm
    """
    x = sp.Symbol('x')

    print(f"\n--- TÍNH M_{order} = max|f^({order})(x)| trên [{a}, {b}] ---")

    # Tính đạo hàm cấp order
    derivative = f_expr_sympy
    for i in range(order):
        derivative = sp.diff(derivative, x)
        print(f"\nf^({i+1})(x) = {derivative}")

    # Chuyển sang numpy function để tính giá trị
    try:
        f_derivative = sp.lambdify(x, derivative, 'numpy')

        # Tính tại nhiều điểm
        x_vals = np.linspace(a, b, num_points)
        y_vals = np.abs(f_derivative(x_vals))

        # Loại bỏ NaN và Inf
        y_vals = y_vals[np.isfinite(y_vals)]

        if len(y_vals) == 0:
            print("⚠ Không thể tính M (giá trị không xác định)")
            return None

        M = np.max(y_vals)
        print(f"\nM_{order} = max|f^({order})(x)| ≈ {M:.6f}")

        return M
    except Exception as e:
        print(f"⚠ Lỗi khi tính M_{order}: {e}")
        return None

def tinh_sai_so_simpson_3_8(f_sympy, a, b, h):
    """
    Sai số Simpson 3/8: |R_n| ≤ (b-a) * h^4 * M4 / 80
    """
    print("\n--- SAI SỐ LÝ THUYẾT (SIMPSON 3/8) ---")
    M4 = tinh_M(f_sympy, 4, a, b)
    if M4:
        sai_so = (b - a) * h**4 * M4 / 80
        print(f"\nCông thức: |I - Iₙ| ≤ (b-a)h⁴M₄/80")
        print(f"         = {b-a} × {h:.6f}⁴ × {M4:.6f} / 80")
        print(f"         ≈ {sai_so:.6e}")
        return sai_so
    return None

def tinh_sai_so_boole(f_sympy, a, b, h):
    """
    Sai số Boole: |R_n| ≤ 2(b-a) * h^6 * M6 / 945
    """
    print("\n--- SAI SỐ LÝ THUYẾT (BOOLE) ---")
    M6 = tinh_M(f_sympy, 6, a, b)
    if M6:
        sai_so = 2 * (b - a) * h**6 * M6 / 945
        print(f"\nCông thức: |I - Iₙ| ≤ 2(b-a)h⁶M₆/945")
        print(f"         = 2 × {b-a} × {h:.6f}⁶ × {M6:.6f} / 945")
        print(f"         ≈ {sai_so:.6e}")
        return sai_so
    return None

def tinh_sai_so_nc5(f_sympy, a, b, h):
    """
    Sai số NC bậc 5: |R_n| ≤ 55(b-a) * h^6 * M6 / 12096
    """
    print("\n--- SAI SỐ LÝ THUYẾT (NEWTON-COTES BẬC 5) ---")
    M6 = tinh_M(f_sympy, 6, a, b)
    if M6:
        sai_so = 55 * (b - a) * h**6 * M6 / 12096
        print(f"\nCông thức: |I - Iₙ| ≤ 55(b-a)h⁶M₆/12096")
        print(f"         = 55 × {b-a} × {h:.6f}⁶ × {M6:.6f} / 12096")
        print(f"         ≈ {sai_so:.6e}")
        return sai_so
    return None

def tinh_n_tu_sai_so_simpson_3_8(epsilon, M4, a, b):
    """
    Từ: (b-a) * h^4 * M4 / 80 < epsilon
    => h < [(80 * epsilon) / ((b-a) * M4)]^(1/4)
    => n > (b-a) / h
    """
    if M4 <= 0:
        return None

    h_max = ((80 * epsilon) / ((b - a) * M4)) ** 0.25
    n_min = (b - a) / h_max
    n = int(np.ceil(n_min))

    # Làm tròn lên bội của 3
    if n % 3 != 0:
        n = ((n // 3) + 1) * 3

    print(f"\nCông thức: ε ≥ (b-a)h⁴M₄/80")
    print(f"=> h ≤ ⁴√(80ε/((b-a)M₄)) = ⁴√(80×{epsilon}/({b-a}×{M4:.6f})) ≈ {h_max:.6f}")
    print(f"=> n ≥ (b-a)/h = {n_min:.2f}")
    print(f"=> n = {n} (làm tròn lên bội của 3)")

    return n

def tinh_n_tu_sai_so_boole(epsilon, M6, a, b):
    """
    Từ: 2(b-a) * h^6 * M6 / 945 < epsilon
    => h < [(945 * epsilon) / (2 * (b-a) * M6)]^(1/6)
    => n > (b-a) / h
    """
    if M6 <= 0:
        return None

    h_max = ((945 * epsilon) / (2 * (b - a) * M6)) ** (1/6)
    n_min = (b - a) / h_max
    n = int(np.ceil(n_min))

    # Làm tròn lên bội của 4
    if n % 4 != 0:
        n = ((n // 4) + 1) * 4

    print(f"\nCông thức: ε ≥ 2(b-a)h⁶M₆/945")
    print(f"=> h ≤ ⁶√(945ε/(2(b-a)M₆)) = ⁶√(945×{epsilon}/(2×{b-a}×{M6:.6f})) ≈ {h_max:.6f}")
    print(f"=> n ≥ (b-a)/h = {n_min:.2f}")
    print(f"=> n = {n} (làm tròn lên bội của 4)")

    return n

def tinh_n_tu_sai_so_nc5(epsilon, M6, a, b):
    """
    Từ: 55(b-a) * h^6 * M6 / 12096 < epsilon
    => h < [(12096 * epsilon) / (55 * (b-a) * M6)]^(1/6)
    => n > (b-a) / h
    """
    if M6 <= 0:
        return None

    h_max = ((12096 * epsilon) / (55 * (b - a) * M6)) ** (1/6)
    n_min = (b - a) / h_max
    n = int(np.ceil(n_min))

    # Làm tròn lên bội của 5
    if n % 5 != 0:
        n = ((n // 5) + 1) * 5

    print(f"\nCông thức: ε ≥ 55(b-a)h⁶M₆/12096")
    print(f"=> h ≤ ⁶√(12096ε/(55(b-a)M₆)) = ⁶√(12096×{epsilon}/(55×{b-a}×{M6:.6f})) ≈ {h_max:.6f}")
    print(f"=> n ≥ (b-a)/h = {n_min:.2f}")
    print(f"=> n = {n} (làm tròn lên bội của 5)")

    return n

def hinh_thang(f, a, b, n):
    """Phương pháp hình thang"""
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    result = h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)
    return result, h, x, y

def simpson(f, a, b, n):
    """Phương pháp Simpson (n phải chẵn)"""
    if n % 2 != 0:
        n += 1
        print(f"⚠ n phải chẵn! Tự động tăng thành {n}")

    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    result = h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))
    return result, h, x, y
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    result = h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)
    return result, h, x, y

def simpson(f, a, b, n):
    if n % 2 != 0:
        n += 1
        print(f"⚠ n phải chẵn! Tự động tăng thành {n}")

    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    result = h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))
    return result, h, x, y

def simpson_3_8(f, a, b, n):
    """
    Simpson 3/8 (Newton-Cotes bậc 3)
    Điều kiện: n phải chia hết cho 3
    Công thức: I ≈ (3h/8)[y0 + 3y1 + 3y2 + 2y3 + 3y4 + 3y5 + ... + y_n]
    Quy luật hệ số: 1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 1
    """
    if n % 3 != 0:
        n = ((n // 3) + 1) * 3
        print(f"⚠ n phải chia hết cho 3! Tự động tăng thành {n}")

    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    # Tính tổng với quy luật hệ số: 1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 1
    result = y[0] + y[-1]  # Điểm đầu và cuối

    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * y[i]  # Các điểm chia hết cho 3
        else:
            result += 3 * y[i]  # Các điểm còn lại

    result *= (3 * h / 8)
    return result, h, x, y

def boole(f, a, b, n):
    """
    Boole's Rule (Newton-Cotes bậc 4)
    Điều kiện: n phải chia hết cho 4
    Công thức: I ≈ (2h/45)[7y0 + 32y1 + 12y2 + 32y3 + 14y4 + 32y5 + ... + 7y_n]
    Quy luật hệ số: 7, 32, 12, 32, 14, 32, 12, 32, 14, ..., 7
    """
    if n % 4 != 0:
        n = ((n // 4) + 1) * 4
        print(f"⚠ n phải chia hết cho 4! Tự động tăng thành {n}")

    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    # Tính tổng với quy luật hệ số
    result = 7 * (y[0] + y[-1])  # Điểm đầu và cuối

    for i in range(1, n):
        if i % 4 == 0:
            result += 14 * y[i]  # Các điểm nối (chia hết cho 4)
        elif i % 2 == 0:
            result += 12 * y[i]  # Các điểm chẵn không chia hết cho 4
        else:
            result += 32 * y[i]  # Các điểm lẻ

    result *= (2 * h / 45)
    return result, h, x, y

def newton_cotes_5(f, a, b, n):
    """
    Newton-Cotes bậc 5
    Điều kiện: n phải chia hết cho 5
    Công thức: I ≈ (5h/288)[19y0 + 75y1 + 50y2 + 50y3 + 75y4 + 19y5 + ...]
    Quy luật: 19, 75, 50, 50, 75, (19+19=38 tại điểm nối), 75, 50, 50, 75, ...
    """
    if n % 5 != 0:
        n = ((n // 5) + 1) * 5
        print(f"⚠ n phải chia hết cho 5! Tự động tăng thành {n}")

    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    # Tính tổng với quy luật hệ số
    result = 19 * (y[0] + y[-1])  # Điểm đầu và cuối

    for i in range(1, n):
        remainder = i % 5
        if remainder == 0:
            result += 38 * y[i]  # Điểm nối (19+19)
        elif remainder == 1 or remainder == 4:
            result += 75 * y[i]  # Vị trí 1 và 4
        else:  # remainder == 2 or remainder == 3
            result += 50 * y[i]  # Vị trí 2 và 3

    result *= (5 * h / 288)
    return result, h, x, y

def tinh_tich_phan():
    clear_screen()
    print("=== TÍNH GẦN ĐÚNG TÍCH PHÂN ===\n")

    print("Nhập hàm số f(x)")
    print("Ví dụ:")
    print("  x^2 + sin(x)")
    print("  e^(-x^2)")
    print("  1/(1+x^2)")
    print("  x^(1/2) + cos(x)    [căn bậc 2]")
    print("  ln(x)               [logarit tự nhiên]")
    print("  log_10(x)           [logarit cơ số 10]")
    print("  log_2(x)            [logarit cơ số 2]")
    print("  2x                  [tự động thành 2*x]")

    f_str = input("\nf(x) = ").strip()

    try:
        f = convert_to_python(f_str)
        f_sympy = convert_to_sympy(f_str)
        print("✓ Đọc hàm thành công!")
        print(f"  SymPy: f(x) = {f_sympy}")
    except Exception as e:
        print(f"✗ Lỗi: {e}")
        return

    a = float(input("\nCận dưới a: "))
    b = float(input("Cận trên b: "))

    print("\n--- CHỌN PHƯƠNG PHÁP ---")
    print("1. Hình thang")
    print("2. Simpson 1/3 (bậc 2)")
    print("3. Simpson 3/8 (bậc 3)")
    print("4. Boole (bậc 4)")
    print("5. Newton-Cotes bậc 5")

    method = input("\nChọn (1/2/3/4/5): ").strip()

    print("\n--- SỐ KHOẢNG CHIA ---")
    print("1. Nhập n trực tiếp")
    print("2. Nhập sai số ε (tự tính n)")

    error_choice = input("Chọn (1/2): ").strip()

    if error_choice == "2":
        epsilon = float(input("Sai số mong muốn ε: "))

        if method == "1":
            # Hình thang: ε ≥ M2*(b-a)*h^2/12 => h ≤ sqrt(12*ε/(M2*(b-a)))
            print("\n" + "="*70)
            M2 = tinh_M(f_sympy, 2, a, b)
            print("="*70)

            if M2 is None:
                print("\n⚠ Không tính được M2 tự động")
                M2 = float(input("Nhập M2 thủ công: "))

            h_max = np.sqrt(12 * epsilon / (M2 * (b - a)))
            n = int(np.ceil((b - a) / h_max))

            print(f"\nCông thức: ε ≥ M₂(b-a)h²/12")
            print(f"=> h ≤ √(12ε/(M₂(b-a))) = √(12×{epsilon}/(M₂×{b-a})) ≈ {h_max:.6f}")
            print(f"=> n ≥ (b-a)/h = {(b-a)/h_max:.2f}")
            print(f"=> n = {n}")

        elif method == "2":
            # Simpson: ε ≥ M4*(b-a)*h^4/180
            print("\n" + "="*70)
            M4 = tinh_M(f_sympy, 4, a, b)
            print("="*70)

            if M4 is None:
                print("\n⚠ Không tính được M4 tự động")
                M4 = float(input("Nhập M4 thủ công: "))

            h_max = (180 * epsilon / (M4 * (b - a))) ** 0.25
            n = int(np.ceil((b - a) / h_max))
            if n % 2 != 0:
                n += 1

            print(f"\nCông thức: ε ≥ M₄(b-a)h⁴/180")
            print(f"=> h ≤ ⁴√(180ε/(M₄(b-a))) = ⁴√(180×{epsilon}/(M₄×{b-a})) ≈ {h_max:.6f}")
            print(f"=> n ≥ (b-a)/h = {(b-a)/h_max:.2f}")
            print(f"=> n = {n} (làm tròn chẵn)")

        elif method == "3":
            # Simpson 3/8: ε ≥ (b-a)*h^4*M4/80
            print("\n" + "="*70)
            M4 = tinh_M(f_sympy, 4, a, b)
            print("="*70)

            if M4 is None:
                print("\n⚠ Không tính được M4 tự động")
                M4 = float(input("Nhập M4 thủ công: "))

            n = tinh_n_tu_sai_so_simpson_3_8(epsilon, M4, a, b)

        elif method == "4":
            # Boole: ε ≥ 2(b-a)*h^6*M6/945
            print("\n" + "="*70)
            M6 = tinh_M(f_sympy, 6, a, b)
            print("="*70)

            if M6 is None:
                print("\n⚠ Không tính được M6 tự động")
                M6 = float(input("Nhập M6 thủ công: "))

            n = tinh_n_tu_sai_so_boole(epsilon, M6, a, b)

        elif method == "5":
            # NC bậc 5: ε ≥ 55(b-a)*h^6*M6/12096
            print("\n" + "="*70)
            M6 = tinh_M(f_sympy, 6, a, b)
            print("="*70)

            if M6 is None:
                print("\n⚠ Không tính được M6 tự động")
                M6 = float(input("Nhập M6 thủ công: "))

            n = tinh_n_tu_sai_so_nc5(epsilon, M6, a, b)
        else:
            print("Phương pháp không hợp lệ")
            n = int(input("Nhập n: "))
    else:
        n = int(input("Nhập n: "))

    print("\n" + "="*80)
    print("KẾT QUẢ TÍNH TOÁN")
    print("="*80)

    try:
        if method == "1":
            result, h, x, y = hinh_thang(f, a, b, n)
            print(f"\n🔹 Phương pháp: HÌNH THANG")
            print(f"🔹 Số khoảng: n = {n}")
            print(f"🔹 Bước: h = {h:.6f}")
            print(f"\n📊 Kết quả: ∫[{a},{b}] f(x)dx ≈ {result:.10f}")

            print(f"\n{'i':>3} {'x_i':>12} {'f(x_i)':>12} {'Hệ số':>8}")
            print("-" * 40)
            for i in range(min(len(x), 11)):
                coeff = 0.5 if (i == 0 or i == len(x)-1) else 1
                print(f"{i:>3} {x[i]:>12.6f} {y[i]:>12.6f} {coeff:>8.1f}")
            if len(x) > 11:
                print("  ...")

            # Tính sai số lý thuyết
            print("\n--- SAI SỐ LÝ THUYẾT ---")
            M2 = tinh_M(f_sympy, 2, a, b)
            if M2:
                sai_so = M2 * (b - a) * h**2 / 12
                print(f"\nCông thức: |I - Iₙ| ≤ M₂(b-a)h²/12")
                print(f"         = {M2:.6f} × {b-a} × {h:.6f}² / 12")
                print(f"         ≈ {sai_so:.6e}")

        elif method == "2":
            result, h, x, y = simpson(f, a, b, n)
            print(f"\n🔹 Phương pháp: SIMPSON")
            print(f"🔹 Số khoảng: n = {n}")
            print(f"🔹 Bước: h = {h:.6f}")
            print(f"\n📊 Kết quả: ∫[{a},{b}] f(x)dx ≈ {result:.10f}")

            print(f"\n{'i':>3} {'x_i':>12} {'f(x_i)':>12} {'Hệ số':>8}")
            print("-" * 40)
            for i in range(min(len(x), 11)):
                if i == 0 or i == len(x) - 1:
                    coeff = 1
                elif i % 2 == 1:
                    coeff = 4
                else:
                    coeff = 2
                print(f"{i:>3} {x[i]:>12.6f} {y[i]:>12.6f} {coeff:>8}")
            if len(x) > 11:
                print("  ...")

            # Sai số
            print("\n--- SAI SỐ LÝ THUYẾT ---")
            M4 = tinh_M(f_sympy, 4, a, b)
            if M4:
                sai_so = M4 * (b - a) * h**4 / 180
                print(f"\nCông thức: |I - Iₙ| ≤ M₄(b-a)h⁴/180")
                print(f"         = {M4:.6f} × {b-a} × {h:.6f}⁴ / 180")
                print(f"         ≈ {sai_so:.6e}")

        elif method == "3":
            # Simpson 3/8
            result, h, x, y = simpson_3_8(f, a, b, n)
            print(f"\n🔹 Phương pháp: SIMPSON 3/8 (Bậc 3)")
            print(f"🔹 Số khoảng: n = {n}")
            print(f"🔹 Bước: h = {h:.6f}")
            print(f"\n📊 Kết quả: ∫[{a},{b}] f(x)dx ≈ {result:.10f}")

            print(f"\n{'i':>3} {'x_i':>12} {'f(x_i)':>12} {'Hệ số':>8}")
            print("-" * 40)
            for i in range(min(len(x), 13)):
                if i == 0 or i == len(x) - 1:
                    coeff = 1
                elif i % 3 == 0:
                    coeff = 2
                else:
                    coeff = 3
                print(f"{i:>3} {x[i]:>12.6f} {y[i]:>12.6f} {coeff:>8}")
            if len(x) > 13:
                print("  ...")

            tinh_sai_so_simpson_3_8(f_sympy, a, b, h)

        elif method == "4":
            # Boole
            result, h, x, y = boole(f, a, b, n)
            print(f"\n🔹 Phương pháp: BOOLE (Bậc 4)")
            print(f"🔹 Số khoảng: n = {n}")
            print(f"🔹 Bước: h = {h:.6f}")
            print(f"\n📊 Kết quả: ∫[{a},{b}] f(x)dx ≈ {result:.10f}")

            print(f"\n{'i':>3} {'x_i':>12} {'f(x_i)':>12} {'Hệ số':>8}")
            print("-" * 40)
            for i in range(min(len(x), 13)):
                if i == 0 or i == len(x) - 1:
                    coeff = 7
                elif i % 4 == 0:
                    coeff = 14
                elif i % 2 == 0:
                    coeff = 12
                else:
                    coeff = 32
                print(f"{i:>3} {x[i]:>12.6f} {y[i]:>12.6f} {coeff:>8}")
            if len(x) > 13:
                print("  ...")

            tinh_sai_so_boole(f_sympy, a, b, h)

        elif method == "5":
            # Newton-Cotes bậc 5
            result, h, x, y = newton_cotes_5(f, a, b, n)
            print(f"\n🔹 Phương pháp: NEWTON-COTES BẬC 5")
            print(f"🔹 Số khoảng: n = {n}")
            print(f"🔹 Bước: h = {h:.6f}")
            print(f"\n📊 Kết quả: ∫[{a},{b}] f(x)dx ≈ {result:.10f}")

            print(f"\n{'i':>3} {'x_i':>12} {'f(x_i)':>12} {'Hệ số':>8}")
            print("-" * 40)
            for i in range(min(len(x), 13)):
                remainder = i % 5
                if i == 0 or i == len(x) - 1:
                    coeff = 19
                elif remainder == 0:
                    coeff = 38
                elif remainder == 1 or remainder == 4:
                    coeff = 75
                else:
                    coeff = 50
                print(f"{i:>3} {x[i]:>12.6f} {y[i]:>12.6f} {coeff:>8}")
            if len(x) > 13:
                print("  ...")

            tinh_sai_so_nc5(f_sympy, a, b, h)

        else:
            print("Phương pháp không hợp lệ!")
            return

        # So sánh
        print("\n--- SO SÁNH VỚI GIÁ TRỊ CHÍNH XÁC ---")
        try:
            exact, _ = quad(f, a, b)
            error = abs(result - exact)
            print(f"Giá trị chính xác: {exact:.10f}")
            print(f"Sai số thực tế:   {error:.6e}")
            print(f"Sai số tương đối:  {error/abs(exact)*100:.6f}%")
        except:
            print("Không tính được giá trị chính xác")

    except Exception as e:
        print(f"\nLỗi: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    while True:
        clear_screen()
        print("╔" + "="*78 + "╗")
        print("║" + " "*15 + "CHƯƠNG TRÌNH TÍNH TOÁN SỐ (Tối ưu)" + " "*28 + "║")
        print("║" + " "*15 + "ĐẠO HÀM VÀ TÍCH PHÂN GẦN ĐÚNG" + " "*33 + "║")
        print("╚" + "="*78 + "╝")

        print("\n--- MENU ---")
        print("1. Tính đạo hàm")
        print("2. Tính tích phân")
        print("0. Thoát")

        choice = input("\nChọn (0/1/2): ").strip()

        if choice == "1":
            tinh_dao_ham()
            input("\nEnter để tiếp tục...")
        elif choice == "2":
            tinh_tich_phan()
            input("\nEnter để tiếp tục...")
        elif choice == "0":
            print("\nCảm ơn!")
            break
        else:
            print("\nLựa chọn không hợp lệ!")
            input("Enter để tiếp tục...")

if __name__ == "__main__":
    main()