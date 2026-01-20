import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Cấu hình hiển thị
rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def kiem_tra_moc_noi_suy(x_data):
    """Kiểm tra điều kiện các mốc nội suy phải khác nhau"""
    if len(x_data) != len(set(x_data)):
        raise ValueError("❌ LỖI: Các mốc nội suy xi phải đôi một khác nhau!")
    
    for i in range(len(x_data)):
        for j in range(i+1, len(x_data)):
            if abs(x_data[i] - x_data[j]) < 1e-10:
                raise ValueError(f"❌ LỖI: Mốc x[{i}] = {x_data[i]} trùng với x[{j}] = {x_data[j]}")
    
    print("✅ Điều kiện 1: Các mốc nội suy đôi một khác nhau")
    return True

def da_thuc_lagrange_co_ban(x, x_data, i):
    """Tính đa thức Lagrange cơ bản Li(x)"""
    n = len(x_data)
    L_i = 1.0
    
    for j in range(n):
        if j != i:
            L_i *= (x - x_data[j]) / (x_data[i] - x_data[j])
    
    return L_i

def kiem_tra_da_thuc_co_ban(x_data):
    """Kiểm tra điều kiện Li(xj) = δij"""
    n = len(x_data)
    print("✅ Điều kiện 3: Kiểm tra tính chất đa thức Lagrange cơ bản:")
    
    for i in range(n):
        for j in range(n):
            L_i_at_xj = da_thuc_lagrange_co_ban(x_data[j], x_data, i)
            expected = 1.0 if i == j else 0.0
            
            if abs(L_i_at_xj - expected) > 1e-10:
                raise ValueError(f"❌ LỖI: L{i}(x{j}) = {L_i_at_xj}, kỳ vọng {expected}")
            
            if i == j:
                print(f"    L{i}(x{i}) = {L_i_at_xj:.6f} ✓")

### BẮT ĐẦU CODE MỚI ###
def format_polynomial(poly, decimal_places):
    """
    Hàm định dạng đa thức (từ np.poly1d) thành chuỗi đẹp mắt.
    Ví dụ: 3.00*x^2 - 2.00*x + 1.00
    """
    terms = []
    coeffs = poly.coeffs
    degree = poly.order
    
    for i, coeff in enumerate(coeffs):
        # Bỏ qua các hệ số quá nhỏ (gần bằng 0)
        if abs(coeff) < 1e-10:
            continue
            
        power = degree - i
        
        # Định dạng hệ số
        term = f"{coeff:.{decimal_places}f}"
        
        # Thêm phần biến x
        if power > 0:
            term += f"⋅x"
        if power > 1:
            term += f"^{power}"
            
        terms.append(term)
    
    # Nối các số hạng, xử lý dấu cộng/trừ
    if not terms:
        return f"0.00"
        
    result = terms[0].replace('+', '')
    for term in terms[1:]:
        if term.startswith('-'):
            result += f" - {term[1:]}"
        else:
            result += f" + {term}"
            
    return result
### KẾT THÚC CODE MỚI ###

def nhap_so_nguyen(prompt, min_val=None, max_val=None):
    """Hàm nhập số nguyên với kiểm tra"""
    while True:
        try:
            value = int(input(prompt))
            if min_val is not None and value < min_val:
                print(f"❌ Giá trị phải >= {min_val}. Vui lòng nhập lại!")
                continue
            if max_val is not None and value > max_val:
                print(f"❌ Giá trị phải <= {max_val}. Vui lòng nhập lại!")
                continue
            return value
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ!")

def nhap_so_thuc(prompt):
    """Hàm nhập số thực với kiểm tra"""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ!")

def nhap_du_lieu():
    """Hàm nhập dữ liệu từ người dùng"""
    print("\n" + "="*70)
    print("NHẬP DỮ LIỆU CHO NỘI SUY LAGRANGE".center(70))
    print("="*70)
    
    # Nhập số lượng điểm
    n_diem = nhap_so_nguyen("\n📊 Nhập số lượng điểm nội suy (≥ 2): ", min_val=2)
    
    x_data = []
    y_data = []
    
    print(f"\n📝 Nhập tọa độ cho {n_diem} điểm:")
    print("-" * 50)
    
    for i in range(n_diem):
        print(f"\n🔹 Điểm thứ {i+1}:")
        
        while True:
            x = nhap_so_thuc(f"    x[{i}] = ")
            if x in x_data:
                print(f"    ❌ Giá trị x = {x} đã tồn tại! Vui lòng nhập giá trị khác.")
            else:
                x_data.append(x)
                break
        
        y = nhap_so_thuc(f"    y[{i}] = ")
        y_data.append(y)
    
    return np.array(x_data), np.array(y_data)

def noi_suy_lagrange(x_data, y_data, x_eval, decimal_places):
    """Hàm nội suy Lagrange chính"""
    
    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)
    
    if len(x_data) != len(y_data):
        raise ValueError("❌ LỖI: Số lượng mốc x và giá trị y phải bằng nhau!")
    
    n = len(x_data) - 1
    
    # Kiểm tra các điều kiện
    print("\n" + "="*70)
    print("KIỂM TRA CÁC ĐIỀU KIỆN".center(70))
    print("="*70 + "\n")
    
    kiem_tra_moc_noi_suy(x_data)
    print(f"✅ Điều kiện 2: Đa thức nội suy có bậc ≤ {n} (đi qua {n+1} điểm)")
    kiem_tra_da_thuc_co_ban(x_data)
    
    ### BẮT ĐẦU CODE MỚI ###
    # Tính toán đa thức rút gọn (dạng a_n*x^n + ... + a_0)
    # np.poly1d([0.0]) tạo đa thức bậc 0 có giá trị 0
    P_n_poly = np.poly1d([0.0]) 
    
    for i in range(n + 1):
        # Tính Li(x) dạng đa thức
        numerator_poly = np.poly1d([1.0])
        denominator_val = 1.0
        
        for j in range(n + 1):
            if i == j:
                continue
            # (x - x_j) -> biểu diễn bằng [1.0, -x_data[j]]
            numerator_poly *= np.poly1d([1.0, -x_data[j]])
            # (x_i - x_j)
            denominator_val *= (x_data[i] - x_data[j])
        
        # Đa thức Li(x) = tử / mẫu
        L_i_poly = numerator_poly / denominator_val
        
        # Pn(x) = Pn(x) + yi * Li(x)
        P_n_poly += L_i_poly * y_data[i]
    
    # Định dạng chuỗi đa thức rút gọn
    poly_string = format_polynomial(P_n_poly, decimal_places)
    ### KẾT THÚC CODE MỚI ###

    # Hàm tính giá trị nội suy (từ hàm P_n(x) gốc, chính xác hơn)
    def P_n(x):
        result = 0.0
        for i in range(len(x_data)):
            result += y_data[i] * da_thuc_lagrange_co_ban(x, x_data, i)
        return result
    
    # In công thức đa thức
    print("\n" + "="*70)
    print("ĐA THỨC NỘI SUY LAGRANGE".center(70))
    print("="*70)
    
    # Hiển thị bảng dữ liệu
    print("\n📋 Bảng dữ liệu đã nhập:")
    print(f"{'i':<5} {'xi':<20} {'yi':<20}")
    print("-" * 45)
    for i in range(len(x_data)):
        print(f"{i:<5} {x_data[i]:<20.{decimal_places}f} {y_data[i]:<20.{decimal_places}f}")
    
    # Tính và hiển thị đa thức Lagrange cơ bản
    print("\n📐 Đa thức Lagrange cơ bản Li(x):")
    print("-" * 70)
    for i in range(len(x_data)):
        tu_so_parts = [f"(x - {x_data[j]:.{decimal_places}f})" for j in range(len(x_data)) if j != i]
        mau_so_parts = [f"({x_data[i]:.{decimal_places}f} - {x_data[j]:.{decimal_places}f})" for j in range(len(x_data)) if j != i]
        
        tu_so = " × ".join(tu_so_parts) if tu_so_parts else "1"
        mau_so = " × ".join(mau_so_parts) if mau_so_parts else "1"
        
        print(f"\nL{i}(x) = {tu_so}")
        print(f"{'':8} {'-' * 60}")
        print(f"{'':8} {mau_so}")
    
    # Công thức tổng quát
    print("\n📝 Công thức tổng quát:")
    print("-" * 70)
    print(f"Pn(x) = Σ yi × Li(x)")
    print(f"      = " + " + ".join([f"({y_data[i]:.{decimal_places}f})×L{i}(x)" for i in range(len(x_data))]))
    
    ### BẮT ĐẦU CODE MỚI ###
    print("\n" + "-" * 70)
    print("🔍 HÀM SỐ CỤ THỂ (SAU KHI RÚT GỌN):".center(70))
    print("-" * 70)
    print(f"\n   Pn(x) = {poly_string}\n")
    ### KẾT THÚC CODE MỚI ###
    
    # Tính giá trị tại các điểm
    if isinstance(x_eval, (int, float)):
        x_eval = [x_eval]
    
    print("\n" + "="*70)
    print("KẾT QUẢ NỘI SUY".center(70))
    print("="*70)
    
    results = []
    for x in x_eval:
        y_interp = P_n(x)
        results.append(y_interp)
        print(f"\n🎯 Tại x = {x:.{decimal_places}f}:")
        print(f"    P({x:.{decimal_places}f}) = {y_interp:.{decimal_places}f}")
    
    # Trả về hàm P_n (để vẽ đồ thị) và kết quả
    return P_n, np.array(results) if len(results) > 1 else results[0]

def ve_do_thi(x_data, y_data, x_test, y_test, P_n, decimal_places):
    """Vẽ đồ thị nội suy"""
    
    print("\n" + "="*70)
    print("VẼ ĐỒ THỊ".center(70))
    print("="*70)
    print("\n📊 Đang tạo đồ thị...")
    
    # Tạo điểm cho đồ thị
    x_min = min(x_data.min(), min(x_test))
    x_max = max(x_data.max(), max(x_test))
    margin = (x_max - x_min) * 0.2
    # Xử lý trường hợp chỉ có 1 điểm test và nó nằm trong khoảng mốc
    if margin == 0: 
        margin = max(abs(x_min), abs(x_max), 1.0) * 0.2
        
    x_plot = np.linspace(x_min - margin, x_max + margin, 300)
    y_plot = [P_n(x) for x in x_plot]
    
    # Vẽ đồ thị
    plt.figure(figsize=(12, 7))
    
    # Đường cong nội suy
    plt.plot(x_plot, y_plot, 'b-', linewidth=2.5, label='Đa thức nội suy Lagrange Pn(x)')
    
    # Các mốc nội suy
    plt.plot(x_data, y_data, 'ro', markersize=12, label='Mốc nội suy (xi, yi)', zorder=5)
    
    # Các điểm kiểm tra
    if isinstance(y_test, (int, float)):
        y_test = [y_test]
    plt.plot(x_test, y_test, 'g^', markersize=14, label='Điểm tính nội suy', zorder=5)
    
    # Thêm nhãn cho các mốc nội suy
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        plt.annotate(f'({x:.{decimal_places}f}, {y:.{decimal_places}f})', 
                     xy=(x, y), xytext=(10, 10),
                     textcoords='offset points', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Thêm nhãn cho các điểm kiểm tra
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        plt.annotate(f'x={x:.{decimal_places}f}\ny={y:.{decimal_places}f}', 
                     xy=(x, y), xytext=(10, -25),
                     textcoords='offset points', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    # Thêm lưới và nhãn
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('x', fontsize=13, fontweight='bold')
    plt.ylabel('y', fontsize=13, fontweight='bold')
    plt.title(f'Đồ thị Nội suy Lagrange (Bậc {len(x_data)-1})', fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.tight_layout()
    
    print("✅ Đồ thị đã được tạo thành công!")
    plt.show()

def main():
    """Hàm chính điều khiển chương trình"""
    
    print("\n" + "🌟"*35)
    print("CHƯƠNG TRÌNH NỘI SUY LAGRANGE".center(70))
    print("🌟"*35)
    
    try:
        # Bước 1: Nhập dữ liệu
        x_data, y_data = nhap_du_lieu()
        
        # Bước 2: Cấu hình độ chính xác
        print("\n" + "="*70)
        print("CẤU HÌNH ĐỘ CHÍNH XÁC".center(70))
        print("="*70)
        decimal_places = nhap_so_nguyen("\n🔢 Số chữ số thập phân sau dấu phẩy (1-10): ", min_val=1, max_val=10)
        
        # Bước 3: Nhập các điểm cần tính nội suy
        print("\n" + "="*70)
        print("ĐIỂM CẦN TÍNH GIÁ TRỊ NỘI SUY".center(70))
        print("="*70)
        
        n_test = nhap_so_nguyen("\n📍 Nhập số điểm cần tính giá trị nội suy: ", min_val=1)
        
        x_test = []
        print(f"\n📝 Nhập {n_test} điểm cần tính:")
        for i in range(n_test):
            x = nhap_so_thuc(f"    Điểm thứ {i+1}, x = ")
            x_test.append(x)
        
        # Bước 4: Thực hiện nội suy
        P_n, y_test = noi_suy_lagrange(x_data, y_data, x_test, decimal_places)
        
        # Bước 5: Vẽ đồ thị
        ve_do_thi(x_data, y_data, x_test, y_test, P_n, decimal_places)
        
        # Bước 6: Tóm tắt kết quả
        print("\n" + "="*70)
        print("TÓM TẮT KẾT QUẢ".center(70))
        print("="*70)
        
        print(f"\n✅ Số mốc nội suy: {len(x_data)}")
        print(f"✅ Bậc đa thức: {len(x_data)-1}")
        print(f"✅ Số điểm đã tính: {len(x_test)}")
        print(f"✅ Độ chính xác: {decimal_places} chữ số thập phân")
        
        print("\n📊 Bảng kết quả chi tiết:")
        print(f"{'STT':<6} {'x':<20} {'P(x)':<20}")
        print("-" * 46)
        
        if isinstance(y_test, (int, float)):
            y_test = [y_test]
        
        for i, (x, y) in enumerate(zip(x_test, y_test)):
            print(f"{i+1:<6} {x:<20.{decimal_places}f} {y:<20.{decimal_places}f}")
        
        # Hỏi người dùng có muốn tiếp tục không
        print("\n" + "="*70)
        tiep_tuc = input("\n🔄 Bạn có muốn chạy lại chương trình với dữ liệu mới? (c/k): ").lower().strip()
        if tiep_tuc == 'c':
            print("\n" * 2)
            main()
        else:
            print("\n" + "="*70)
            print("CẢM ƠN BẠN ĐÃ SỬ DỤNG CHƯƠNG TRÌNH!".center(70))
            print("="*70)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Chương trình đã bị dừng bởi người dùng.")
    except Exception as e:
        print(f"\n\n❌ Đã xảy ra lỗi: {e}")
        print("\n💡 Vui lòng thử lại!")

# Chạy chương trình
if __name__ == "__main__":
    main()