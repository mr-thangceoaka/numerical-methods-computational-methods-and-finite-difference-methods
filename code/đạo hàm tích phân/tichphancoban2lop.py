import numpy as np
from scipy.integrate import dblquad
import sympy as sp

def clear_screen():
    print("\n" + "="*80 + "\n")

# ============================================================================
# PHẦN XỬ LÝ HÀM SỐ 2 BIẾN - f(x,y)
# ============================================================================

def convert_to_python_2d(expr_str):
    """
    Chuyển đổi biểu thức toán học 2 biến f(x,y) sang Python
    - Dùng ^ cho lũy thừa
    - Dùng e cho số Euler
    - ln(x) = logarit tự nhiên
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

    # Nếu đã là lambda
    if expr_str.startswith('lambda'):
        expr_str = expr_str.replace('^', '**')
        return eval(expr_str)

    # Xử lý log_<n>
    expr_str = expr_str.replace('log_10', '__LOG10__')
    expr_str = expr_str.replace('log_2', '__LOG2__')
    expr_str = expr_str.replace('log_e', '__LOGE__')

    # Tự động thêm *
    expr_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr_str)
    expr_str = re.sub(r'\)(\d)', r')*\1', expr_str)
    expr_str = re.sub(r'\)([a-zA-Z])', r')*\1', expr_str)

    # Thay thế hàm
    expr_str = expr_str.replace('ln', '__LN__')
    expr_str = expr_str.replace('sin', '__SIN__')
    expr_str = expr_str.replace('cos', '__COS__')
    expr_str = expr_str.replace('tan', '__TAN__')

    # Thay ^ thành **
    expr_str = expr_str.replace('^', '**')

    # Xử lý e
    expr_str = re.sub(r'\be\b', '__E__', expr_str)

    # Thay placeholder
    replacements = {
        '__LN__': 'np.log',
        '__LOG10__': 'np.log10',
        '__LOG2__': 'np.log2',
        '__LOGE__': 'np.log',
        '__SIN__': 'np.sin',
        '__COS__': 'np.cos',
        '__TAN__': 'np.tan',
        '__E__': 'np.e',
    }

    for placeholder, func in replacements.items():
        expr_str = expr_str.replace(placeholder, func)

    # Tạo lambda function với 2 biến
    try:
        f = eval(f'lambda x, y: {expr_str}')
        # Test
        test_val = f(2.0, 2.0)
        if not np.isfinite(test_val):
            raise ValueError("Hàm cho giá trị không xác định (inf/nan)")
        return f
    except SyntaxError as e:
        raise ValueError(f"Lỗi cú pháp: {e}\nChuỗi: lambda x, y: {expr_str}")
    except Exception as e:
        raise ValueError(f"Lỗi: {e}\nChuỗi: lambda x, y: {expr_str}")

def convert_to_sympy_2d(expr_str):
    """Chuyển biểu thức 2 biến thành SymPy để tính đạo hàm riêng"""
    import re

    expr_str = expr_str.strip()

    # Bỏ lambda nếu có
    if expr_str.startswith('lambda x, y:'):
        expr_str = expr_str[12:].strip()

    # Tự động thêm *
    expr_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr_str)
    expr_str = re.sub(r'\)(\d)', r')*\1', expr_str)
    expr_str = re.sub(r'\)([a-zA-Z])', r')*\1', expr_str)

    # Xử lý log
    expr_str = re.sub(r'log_(\d+)\(([^)]+)\)', r'log(\2, \1)', expr_str)
    expr_str = expr_str.replace('ln(', 'log(')

    # Thay ^ thành **
    expr_str = expr_str.replace('^', '**')

    # Xử lý e
    expr_str = re.sub(r'\be\b', 'E', expr_str)

    # Thay np.
    expr_str = expr_str.replace('np.', '')

    try:
        x, y = sp.symbols('x y')
        return sp.sympify(expr_str)
    except:
        try:
            return sp.parse_expr(expr_str, transformations='all')
        except Exception as e:
            raise ValueError(f"Không thể chuyển sang SymPy: {e}\nBiểu thức: {expr_str}")

# ============================================================================
# PHẦN TÍCH PHÂN 2 LỚP
# ============================================================================

def hinh_thang_2d(f, a, b, c, d, n, m):
    """
    Tích phân 2 lớp bằng phương pháp Hình thang 2D
    ∬[a,b]×[c,d] f(x,y) dxdy

    Quy luật trọng số:
    - 4 góc: 1
    - Biên (không phải góc): 2
    - Điểm trong: 4

    Công thức: I ≈ (hk/4) Σ Σ w_ij f(x_i, y_j)
    """
    h = (b - a) / n
    k = (d - c) / m

    # Tạo lưới điểm
    x = np.linspace(a, b, n+1)
    y = np.linspace(c, d, m+1)
    X, Y = np.meshgrid(x, y)

    # Tính giá trị hàm tại các điểm lưới
    Z = f(X, Y)

    # Tính tổng với trọng số
    result = 0.0

    # 4 góc (trọng số 1)
    result += Z[0, 0] + Z[0, -1] + Z[-1, 0] + Z[-1, -1]

    # Biên trên và dưới (không tính góc, trọng số 2)
    result += 2 * (np.sum(Z[0, 1:-1]) + np.sum(Z[-1, 1:-1]))

    # Biên trái và phải (không tính góc, trọng số 2)
    result += 2 * (np.sum(Z[1:-1, 0]) + np.sum(Z[1:-1, -1]))

    # Các điểm trong (trọng số 4)
    result += 4 * np.sum(Z[1:-1, 1:-1])

    result *= (h * k / 4)

    return result, h, k, X, Y, Z

def simpson_2d(f, a, b, c, d, n, m):
    """
    Tích phân 2 lớp bằng phương pháp Simpson 2D
    ∬[a,b]×[c,d] f(x,y) dxdy

    Điều kiện: n và m đều phải CHẴN

    Quy luật trọng số (nhân tensor product):
    - Góc: 1×1 = 1
    - Biên lẻ: 4×1 hoặc 1×4 = 4
    - Biên chẵn: 2×1 hoặc 1×2 = 2
    - Trong (lẻ, lẻ): 4×4 = 16
    - Trong (chẵn, chẵn): 2×2 = 4
    - Trong (lẻ, chẵn) hoặc (chẵn, lẻ): 4×2 hoặc 2×4 = 8

    Công thức: I ≈ (hk/9) Σ Σ w_ij f(x_i, y_j)
    """
    if n % 2 != 0:
        n += 1
        print(f"⚠ n phải chẵn! Tự động tăng thành {n}")
    if m % 2 != 0:
        m += 1
        print(f"⚠ m phải chẵn! Tự động tăng thành {m}")

    h = (b - a) / n
    k = (d - c) / m

    # Tạo lưới điểm
    x = np.linspace(a, b, n+1)
    y = np.linspace(c, d, m+1)
    X, Y = np.meshgrid(x, y)

    # Tính giá trị hàm tại các điểm lưới
    Z = f(X, Y)

    # Tạo ma trận trọng số Simpson 1D
    def simpson_weights_1d(n):
        w = np.ones(n+1)
        w[1:-1:2] = 4  # Các chỉ số lẻ
        w[2:-1:2] = 2  # Các chỉ số chẵn (không phải đầu/cuối)
        return w

    w_x = simpson_weights_1d(n)
    w_y = simpson_weights_1d(m)

    # Tạo ma trận trọng số 2D (tensor product)
    W = np.outer(w_y, w_x)  # outer product

    # Tính tích phân
    result = (h * k / 9) * np.sum(W * Z)

    return result, h, k, X, Y, Z, W

# ============================================================================
# PHẦN TÍNH M (ĐẠO HÀM RIÊNG)
# ============================================================================

def tinh_M_dao_ham_rieng(f_sympy, bien, bac, a, b, c, d, num_points=100):
    """
    Tính M = max|∂ⁿf/∂biếnⁿ| trên [a,b]×[c,d]
    bien: 'x' hoặc 'y'
    bac: 2 hoặc 4
    """
    x_sym, y_sym = sp.symbols('x y')

    print(f"\n--- TÍNH M_{bien*bac} = max|∂^{bac}f/∂{bien}^{bac}| trên [{a},{b}]×[{c},{d}] ---")

    # Tính đạo hàm riêng
    derivative = f_sympy
    for i in range(bac):
        if bien == 'x':
            derivative = sp.diff(derivative, x_sym)
        else:
            derivative = sp.diff(derivative, y_sym)
        print(f"\n∂^{i+1}f/∂{bien}^{i+1} = {derivative}")

    # Chuyển sang numpy function
    try:
        f_derivative = sp.lambdify((x_sym, y_sym), derivative, 'numpy')

        # Tính tại nhiều điểm
        x_vals = np.linspace(a, b, num_points)
        y_vals = np.linspace(c, d, num_points)
        X, Y = np.meshgrid(x_vals, y_vals)

        Z_vals = np.abs(f_derivative(X, Y))

        # Loại bỏ NaN và Inf
        Z_vals = Z_vals[np.isfinite(Z_vals)]

        if len(Z_vals) == 0:
            print("⚠ Không thể tính M (giá trị không xác định)")
            return None

        M = np.max(Z_vals)
        print(f"\nM_{bien*bac} = max|∂^{bac}f/∂{bien}^{bac}| ≈ {M:.6f}")

        return M
    except Exception as e:
        print(f"⚠ Lỗi khi tính M_{bien*bac}: {e}")
        return None

def tinh_sai_so_hinh_thang_2d(f_sympy, a, b, c, d, h, k):
    """
    Sai số Hình thang 2D: |E| ≤ (b-a)(d-c)/12 × (h²M_xx + k²M_yy)
    """
    print("\n" + "="*80)
    print("SAI SỐ LÝ THUYẾT (HÌNH THANG 2D)")
    print("="*80)

    # Tính M_xx
    M_xx = tinh_M_dao_ham_rieng(f_sympy, 'x', 2, a, b, c, d)

    # Tính M_yy
    M_yy = tinh_M_dao_ham_rieng(f_sympy, 'y', 2, a, b, c, d)

    if M_xx and M_yy:
        sai_so = (b - a) * (d - c) / 12 * (h**2 * M_xx + k**2 * M_yy)
        print("\n" + "="*80)
        print(f"Công thức: |E| ≤ (b-a)(d-c)/12 × (h²M_xx + k²M_yy)")
        print(f"         = {b-a} × {d-c} / 12 × ({h:.6f}² × {M_xx:.6f} + {k:.6f}² × {M_yy:.6f})")
        print(f"         ≈ {sai_so:.6e}")
        print("="*80)
        return sai_so
    return None

def tinh_sai_so_simpson_2d(f_sympy, a, b, c, d, h, k):
    """
    Sai số Simpson 2D: |E| ≤ (b-a)(d-c)/180 × (h⁴M_xxxx + k⁴M_yyyy)
    """
    print("\n" + "="*80)
    print("SAI SỐ LÝ THUYẾT (SIMPSON 2D)")
    print("="*80)

    # Tính M_xxxx
    M_xxxx = tinh_M_dao_ham_rieng(f_sympy, 'x', 4, a, b, c, d)

    # Tính M_yyyy
    M_yyyy = tinh_M_dao_ham_rieng(f_sympy, 'y', 4, a, b, c, d)

    if M_xxxx and M_yyyy:
        sai_so = (b - a) * (d - c) / 180 * (h**4 * M_xxxx + k**4 * M_yyyy)
        print("\n" + "="*80)
        print(f"Công thức: |E| ≤ (b-a)(d-c)/180 × (h⁴M_xxxx + k⁴M_yyyy)")
        print(f"         = {b-a} × {d-c} / 180 × ({h:.6f}⁴ × {M_xxxx:.6f} + {k:.6f}⁴ × {M_yyyy:.6f})")
        print(f"         ≈ {sai_so:.6e}")
        print("="*80)
        return sai_so
    return None

# ============================================================================
# PHẦN TÍNH n, m TỪ SAI SỐ
# ============================================================================

def tinh_n_m_tu_sai_so_hinh_thang_2d(epsilon, M_xx, M_yy, a, b, c, d):
    """
    Từ: (b-a)(d-c)/12 × (h²M_xx + k²M_yy) < ε

    Nếu chọn h = k (lưới vuông):
    => h² < 12ε / ((b-a)(d-c)(M_xx + M_yy))
    => n = m ≥ max((b-a)/h, (d-c)/k)
    """
    if M_xx <= 0 or M_yy <= 0:
        return None, None

    # Giả sử h ≈ k (lưới vuông đều)
    # h = (b-a)/n, k = (d-c)/m
    # Nếu chọn n/m = (b-a)/(d-c) thì h = k

    # Ước tính h = k
    h_max_squared = 12 * epsilon / ((b - a) * (d - c) * (M_xx + M_yy))

    if h_max_squared <= 0:
        return None, None

    h_max = np.sqrt(h_max_squared)

    n = int(np.ceil((b - a) / h_max))
    m = int(np.ceil((d - c) / h_max))

    print(f"\nCông thức: ε ≥ (b-a)(d-c)/12 × (h²M_xx + k²M_yy)")
    print(f"Giả sử h ≈ k (lưới vuông đều):")
    print(f"=> h² ≈ 12ε/((b-a)(d-c)(M_xx + M_yy))")
    print(f"=> h ≤ √({12*epsilon:.6f}/({(b-a)*(d-c):.4f}×({M_xx + M_yy:.6f}))) ≈ {h_max:.6f}")
    print(f"=> n ≥ (b-a)/h = {(b-a)/h_max:.2f} → n = {n}")
    print(f"=> m ≥ (d-c)/k = {(d-c)/h_max:.2f} → m = {m}")

    return n, m

def tinh_n_m_tu_sai_so_simpson_2d(epsilon, M_xxxx, M_yyyy, a, b, c, d):
    """
    Từ: (b-a)(d-c)/180 × (h⁴M_xxxx + k⁴M_yyyy) < ε

    Nếu chọn h = k:
    => h⁴ < 180ε / ((b-a)(d-c)(M_xxxx + M_yyyy))
    """
    if M_xxxx <= 0 or M_yyyy <= 0:
        return None, None

    h_max_4th = 180 * epsilon / ((b - a) * (d - c) * (M_xxxx + M_yyyy))

    if h_max_4th <= 0:
        return None, None

    h_max = h_max_4th ** 0.25

    n = int(np.ceil((b - a) / h_max))
    m = int(np.ceil((d - c) / h_max))

    # Simpson cần n, m chẵn
    if n % 2 != 0:
        n += 1
    if m % 2 != 0:
        m += 1

    print(f"\nCông thức: ε ≥ (b-a)(d-c)/180 × (h⁴M_xxxx + k⁴M_yyyy)")
    print(f"Giả sử h ≈ k:")
    print(f"=> h⁴ ≈ 180ε/((b-a)(d-c)(M_xxxx + M_yyyy))")
    print(f"=> h ≤ ⁴√({180*epsilon:.6f}/({(b-a)*(d-c):.4f}×{M_xxxx + M_yyyy:.6f})) ≈ {h_max:.6f}")
    print(f"=> n ≥ {(b-a)/h_max:.2f}, m ≥ {(d-c)/h_max:.2f}")
    print(f"=> n = {n}, m = {m} (làm tròn chẵn)")

    return n, m

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def tinh_tich_phan_2d():
    clear_screen()
    print("╔" + "="*78 + "╗")
    print("║" + " "*18 + "TÍCH PHÂN 2 LỚP (TÍCH PHÂN BỘI)" + " "*28 + "║")
    print("║" + " "*25 + "∬[a,b]×[c,d] f(x,y) dxdy" + " "*29 + "║")
    print("╚" + "="*78 + "╝")

    print("\nNhập hàm số f(x,y)")
    print("Ví dụ:")
    print("  x^2 + y^2")
    print("  sin(x)*cos(y)")
    print("  e^(x+y)")
    print("  x*y^2")
    print("  ln(x+y+1)")
    print("  2x*y                [tự động thành 2*x*y]")

    f_str = input("\nf(x,y) = ").strip()

    try:
        f = convert_to_python_2d(f_str)
        f_sympy = convert_to_sympy_2d(f_str)
        print("✓ Đọc hàm thành công!")
        print(f"  SymPy: f(x,y) = {f_sympy}")
    except Exception as e:
        print(f"✗ Lỗi: {e}")
        return

    print("\n--- MIỀN TÍCH PHÂN: [a,b] × [c,d] ---")
    a = float(input("Cận dưới x: a = "))
    b = float(input("Cận trên x:  b = "))
    c = float(input("Cận dưới y: c = "))
    d = float(input("Cận trên y:  d = "))

    print("\n--- CHỌN PHƯƠNG PHÁP ---")
    print("1. Hình thang 2D")
    print("2. Simpson 2D (1/3)")

    method = input("\nChọn (1/2): ").strip()

    print("\n--- SỐ KHOẢNG CHIA ---")
    print("1. Nhập n, m trực tiếp")
    print("2. Nhập sai số ε (tự tính n, m)")

    error_choice = input("Chọn (1/2): ").strip()

    if error_choice == "2":
        epsilon = float(input("\nSai số mong muốn ε: "))

        if method == "1":
            # Hình thang 2D
            print("\n" + "="*80)
            M_xx = tinh_M_dao_ham_rieng(f_sympy, 'x', 2, a, b, c, d)
            M_yy = tinh_M_dao_ham_rieng(f_sympy, 'y', 2, a, b, c, d)
            print("="*80)

            if M_xx is None or M_yy is None:
                print("\n⚠ Không tính được M tự động")
                M_xx = float(input("Nhập M_xx thủ công: "))
                M_yy = float(input("Nhập M_yy thủ công: "))

            n, m = tinh_n_m_tu_sai_so_hinh_thang_2d(epsilon, M_xx, M_yy, a, b, c, d)

        else:
            # Simpson 2D
            print("\n" + "="*80)
            M_xxxx = tinh_M_dao_ham_rieng(f_sympy, 'x', 4, a, b, c, d)
            M_yyyy = tinh_M_dao_ham_rieng(f_sympy, 'y', 4, a, b, c, d)
            print("="*80)

            if M_xxxx is None or M_yyyy is None:
                print("\n⚠ Không tính được M tự động")
                M_xxxx = float(input("Nhập M_xxxx thủ công: "))
                M_yyyy = float(input("Nhập M_yyyy thủ công: "))

            n, m = tinh_n_m_tu_sai_so_simpson_2d(epsilon, M_xxxx, M_yyyy, a, b, c, d)
    else:
        n = int(input("\nNhập số khoảng chia theo x (n): "))
        m = int(input("Nhập số khoảng chia theo y (m): "))

    print("\n" + "="*80)
    print("KẾT QUẢ TÍNH TOÁN")
    print("="*80)

    try:
        if method == "1":
            # Hình thang 2D
            result, h, k, X, Y, Z = hinh_thang_2d(f, a, b, c, d, n, m)

            print(f"\n🔹 Phương pháp: HÌNH THANG 2D")
            print(f"🔹 Lưới: {n} × {m} = {(n+1)*(m+1)} điểm")
            print(f"🔹 Bước: h = {h:.6f}, k = {k:.6f}")
            print(f"\n📊 Kết quả: ∬[{a},{b}]×[{c},{d}] f(x,y) dxdy ≈ {result:.10f}")

            # Hiển thị một số giá trị lưới
            print(f"\n--- MỘT SỐ GIÁ TRỊ TRÊN LƯỚI ---")
            print(f"{'i':>3} {'j':>3} {'x_i':>10} {'y_j':>10} {'f(x_i,y_j)':>12} {'Trọng số':>10}")
            print("-" * 60)

            # Hiển thị 4 góc
            corners = [(0, 0), (0, m), (n, 0), (n, m)]
            for i, j in corners:
                w = 1
                print(f"{i:>3} {j:>3} {X[j,i]:>10.4f} {Y[j,i]:>10.4f} {Z[j,i]:>12.6f} {w:>10} (góc)")

            # Hiển thị vài điểm biên
            if n > 2 and m > 2:
                print(f"{0:>3} {1:>3} {X[1,0]:>10.4f} {Y[1,0]:>10.4f} {Z[1,0]:>12.6f} {2:>10} (biên)")
                print(f"{1:>3} {1:>3} {X[1,1]:>10.4f} {Y[1,1]:>10.4f} {Z[1,1]:>12.6f} {4:>10} (trong)")

            # Tính sai số lý thuyết
            tinh_sai_so_hinh_thang_2d(f_sympy, a, b, c, d, h, k)

        else:
            # Simpson 2D
            result, h, k, X, Y, Z, W = simpson_2d(f, a, b, c, d, n, m)

            print(f"\n🔹 Phương pháp: SIMPSON 2D (1/3)")
            print(f"🔹 Lưới: {n} × {m} = {(n+1)*(m+1)} điểm")
            print(f"🔹 Bước: h = {h:.6f}, k = {k:.6f}")
            print(f"\n📊 Kết quả: ∬[{a},{b}]×[{c},{d}] f(x,y) dxdy ≈ {result:.10f}")

            # Hiển thị một số giá trị
            print(f"\n--- MỘT SỐ GIÁ TRỊ TRÊN LƯỚI ---")
            print(f"{'i':>3} {'j':>3} {'x_i':>10} {'y_j':>10} {'f(x_i,y_j)':>12} {'Trọng số':>10}")
            print("-" * 60)

            # 4 góc
            corners = [(0, 0), (0, m), (n, 0), (n, m)]
            for i, j in corners:
                print(f"{i:>3} {j:>3} {X[j,i]:>10.4f} {Y[j,i]:>10.4f} {Z[j,i]:>12.6f} {int(W[j,i]):>10} (góc)")

            # Vài điểm khác
            if n > 2 and m > 2:
                print(f"{1:>3} {1:>3} {X[1,1]:>10.4f} {Y[1,1]:>10.4f} {Z[1,1]:>12.6f} {int(W[1,1]):>10} (lẻ,lẻ)")
                if n > 3 and m > 3:
                    print(f"{2:>3} {2:>3} {X[2,2]:>10.4f} {Y[2,2]:>10.4f} {Z[2,2]:>12.6f} {int(W[2,2]):>10} (chẵn,chẵn)")

            # Hiển thị ma trận trọng số (nếu nhỏ)
            if n <= 6 and m <= 6:
                print(f"\n--- MA TRẬN TRỌNG SỐ ({m+1}×{n+1}) ---")
                print(W.astype(int))

            # Tính sai số lý thuyết
            tinh_sai_so_simpson_2d(f_sympy, a, b, c, d, h, k)

        # So sánh với SciPy
        print("\n--- SO SÁNH VỚI GIÁ TRỊ CHÍNH XÁC ---")
        try:
            # dblquad nhận f(y, x) - chú ý thứ tự!
            exact, _ = dblquad(lambda y_val, x_val: f(x_val, y_val), a, b, c, d)
            error = abs(result - exact)
            print(f"Giá trị chính xác (SciPy): {exact:.10f}")
            print(f"Sai số thực tế:            {error:.6e}")
            if abs(exact) > 1e-10:
                print(f"Sai số tương đối:          {error/abs(exact)*100:.6f}%")
        except Exception as e:
            print(f"Không tính được giá trị chính xác: {e}")

    except Exception as e:
        print(f"\nLỗi: {e}")

def main():
    while True:
        clear_screen()
        print("╔" + "="*78 + "╗")
        print("║" + " "*15 + "CHƯƠNG TRÌNH TÍCH PHÂN 2 LỚP (TÍCH PHÂN BỘI)" + " "*18 + "║")
        print("║" + " "*25 + "∬[a,b]×[c,d] f(x,y) dxdy" + " "*29 + "║")
        print("╚" + "="*78 + "╝")

        print("\n--- MENU ---")
        print("1. Tính tích phân 2 lớp")
        print("0. Thoát")

        choice = input("\nChọn (0/1): ").strip()

        if choice == "1":
            tinh_tich_phan_2d()
            input("\nEnter để tiếp tục...")
        elif choice == "0":
            print("\nCảm ơn!")
            break
        else:
            print("\nLựa chọn không hợp lệ!")
            input("Enter để tiếp tục...")

if __name__ == "__main__":
    main()