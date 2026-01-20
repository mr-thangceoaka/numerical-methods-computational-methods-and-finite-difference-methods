"""
Spline Interpolation Solver - ULTIMATE VERSION
Phiên bản hoàn hảo với đầy đủ điều kiện biên
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import warnings
import os
warnings.filterwarnings('ignore')

# Thiết lập font cho matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class SplineSolverUltimate:
    """Lớp giải bài toán Spline với đầy đủ điều kiện biên"""

    def __init__(self, x_data, y_data, decimal_places=4):
        """Khởi tạo solver"""
        self.x = np.array(x_data, dtype=float)
        self.y = np.array(y_data, dtype=float)
        self.n = len(self.x)
        self.decimal_places = decimal_places

        if self.n < 2:
            raise ValueError("Cần ít nhất 2 điểm dữ liệu!")

        if len(self.x) != len(self.y):
            raise ValueError("Số lượng điểm x và y phải bằng nhau!")

        # Sắp xếp theo x
        sorted_indices = np.argsort(self.x)
        self.x = self.x[sorted_indices]
        self.y = self.y[sorted_indices]

        # Tính khoảng cách h giữa các điểm
        self.h = np.diff(self.x)

        self.coefficients = None
        self.spline_type = None

    def linear_spline(self):
        """Spline tuyến tính (cấp 1)"""
        self.spline_type = "Linear Spline (Cap 1)"
        coeffs = []

        for k in range(self.n - 1):
            a = (self.y[k+1] - self.y[k]) / self.h[k]
            b = self.y[k] - a * self.x[k]

            coeffs.append({
                'interval': f'[{self.x[k]:.{self.decimal_places}f}, {self.x[k+1]:.{self.decimal_places}f}]',
                'a': round(a, self.decimal_places),
                'b': round(b, self.decimal_places),
                'formula': f'S_{k}(x) = {round(a, self.decimal_places)}x + {round(b, self.decimal_places)}'
            })

        self.coefficients = coeffs
        return coeffs

    def quadratic_spline(self, m0=None):
        """Spline bậc 2 với điều kiện biên S'(x_0) = m0"""
        if m0 is None:
            m0 = 0.0

        self.spline_type = f"Quadratic Spline (Cap 2, S'({self.x[0]})={m0})"

        m = np.zeros(self.n)
        m[0] = m0

        for k in range(self.n - 1):
            m[k+1] = 2 * (self.y[k+1] - self.y[k]) / self.h[k] - m[k]

        coeffs = []

        for k in range(self.n - 1):
            a = (m[k+1] - m[k]) / (2 * self.h[k])
            b = m[k] - 2 * a * self.x[k]
            c = self.y[k] + a * self.x[k]**2 - m[k] * self.x[k]

            coeffs.append({
                'interval': f'[{self.x[k]:.{self.decimal_places}f}, {self.x[k+1]:.{self.decimal_places}f}]',
                'a': round(a, self.decimal_places),
                'b': round(b, self.decimal_places),
                'c': round(c, self.decimal_places),
                'formula': f'S_{k}(x) = {round(a, self.decimal_places)}x^2 + {round(b, self.decimal_places)}x + {round(c, self.decimal_places)}'
            })

        self.coefficients = coeffs
        return coeffs

    def cubic_spline_natural(self):
        """Natural Cubic Spline: S''(x_0) = S''(x_n) = 0"""
        self.spline_type = "Natural Cubic Spline (S''(x0)=S''(xn)=0)"

        cs = CubicSpline(self.x, self.y, bc_type='natural')

        coeffs = []
        for k in range(self.n - 1):
            c = cs.c[:, k]

            coeffs.append({
                'interval': f'[{self.x[k]:.{self.decimal_places}f}, {self.x[k+1]:.{self.decimal_places}f}]',
                'a': round(c[0], self.decimal_places),
                'b': round(c[1], self.decimal_places),
                'c': round(c[2], self.decimal_places),
                'd': round(c[3], self.decimal_places),
                'formula': f'S_{k}(x) = {round(c[0], self.decimal_places)}(x-{self.x[k]})^3 + {round(c[1], self.decimal_places)}(x-{self.x[k]})^2 + {round(c[2], self.decimal_places)}(x-{self.x[k]}) + {round(c[3], self.decimal_places)}'
            })

        self.coefficients = coeffs
        self.cs_func = cs
        return coeffs

    def cubic_spline_clamped(self, fp0, fpn):
        """Clamped Cubic Spline: S'(x_0) = fp0, S'(x_n) = fpn"""
        self.spline_type = f"Clamped Cubic Spline (S'({self.x[0]})={fp0}, S'({self.x[-1]})={fpn})"

        cs = CubicSpline(self.x, self.y, bc_type=((1, fp0), (1, fpn)))

        coeffs = []
        for k in range(self.n - 1):
            c = cs.c[:, k]

            coeffs.append({
                'interval': f'[{self.x[k]:.{self.decimal_places}f}, {self.x[k+1]:.{self.decimal_places}f}]',
                'a': round(c[0], self.decimal_places),
                'b': round(c[1], self.decimal_places),
                'c': round(c[2], self.decimal_places),
                'd': round(c[3], self.decimal_places),
                'formula': f'S_{k}(x) = {round(c[0], self.decimal_places)}(x-{self.x[k]})^3 + {round(c[1], self.decimal_places)}(x-{self.x[k]})^2 + {round(c[2], self.decimal_places)}(x-{self.x[k]}) + {round(c[3], self.decimal_places)}'
            })

        self.coefficients = coeffs
        self.cs_func = cs
        return coeffs

    def cubic_spline_second_derivative(self, fpp0, fppn):
        """
        Cubic Spline với điều kiện biên đạo hàm cấp 2
        S''(x_0) = fpp0, S''(x_n) = fppn

        ĐÂY LÀ TÍNH NĂNG MỚI!
        """
        self.spline_type = f"Cubic Spline (S''({self.x[0]})={fpp0}, S''({self.x[-1]})={fppn})"

        # bc_type=((2, value), (2, value)) cho điều kiện biên đạo hàm cấp 2
        cs = CubicSpline(self.x, self.y, bc_type=((2, fpp0), (2, fppn)))

        coeffs = []
        for k in range(self.n - 1):
            c = cs.c[:, k]

            coeffs.append({
                'interval': f'[{self.x[k]:.{self.decimal_places}f}, {self.x[k+1]:.{self.decimal_places}f}]',
                'a': round(c[0], self.decimal_places),
                'b': round(c[1], self.decimal_places),
                'c': round(c[2], self.decimal_places),
                'd': round(c[3], self.decimal_places),
                'formula': f'S_{k}(x) = {round(c[0], self.decimal_places)}(x-{self.x[k]})^3 + {round(c[1], self.decimal_places)}(x-{self.x[k]})^2 + {round(c[2], self.decimal_places)}(x-{self.x[k]}) + {round(c[3], self.decimal_places)}'
            })

        self.coefficients = coeffs
        self.cs_func = cs
        return coeffs

    def cubic_spline_mixed(self, bc_left_type, bc_left_value, bc_right_type, bc_right_value):
        """
        Cubic Spline với điều kiện biên hỗn hợp

        Parameters:
        -----------
        bc_left_type : int
            1 = S'(x_0), 2 = S''(x_0)
        bc_left_value : float
            Giá trị điều kiện biên trái
        bc_right_type : int
            1 = S'(x_n), 2 = S''(x_n)
        bc_right_value : float
            Giá trị điều kiện biên phải
        """
        left_str = f"S'({self.x[0]})={bc_left_value}" if bc_left_type == 1 else f"S''({self.x[0]})={bc_left_value}"
        right_str = f"S'({self.x[-1]})={bc_right_value}" if bc_right_type == 1 else f"S''({self.x[-1]})={bc_right_value}"

        self.spline_type = f"Cubic Spline ({left_str}, {right_str})"

        cs = CubicSpline(self.x, self.y, bc_type=((bc_left_type, bc_left_value), (bc_right_type, bc_right_value)))

        coeffs = []
        for k in range(self.n - 1):
            c = cs.c[:, k]

            coeffs.append({
                'interval': f'[{self.x[k]:.{self.decimal_places}f}, {self.x[k+1]:.{self.decimal_places}f}]',
                'a': round(c[0], self.decimal_places),
                'b': round(c[1], self.decimal_places),
                'c': round(c[2], self.decimal_places),
                'd': round(c[3], self.decimal_places),
                'formula': f'S_{k}(x) = {round(c[0], self.decimal_places)}(x-{self.x[k]})^3 + {round(c[1], self.decimal_places)}(x-{self.x[k]})^2 + {round(c[2], self.decimal_places)}(x-{self.x[k]}) + {round(c[3], self.decimal_places)}'
            })

        self.coefficients = coeffs
        self.cs_func = cs
        return coeffs

    def evaluate(self, x_eval):
        """Tính giá trị của hàm spline"""
        if hasattr(self, 'cs_func'):
            return self.cs_func(x_eval)
        else:
            y_eval = np.zeros_like(x_eval)

            for i, x in enumerate(x_eval):
                k = np.searchsorted(self.x[1:], x)
                k = min(k, self.n - 2)

                coeff = self.coefficients[k]

                if 'a' in coeff and 'b' in coeff and 'c' not in coeff:
                    y_eval[i] = coeff['a'] * x + coeff['b']
                elif 'a' in coeff and 'b' in coeff and 'c' in coeff:
                    y_eval[i] = coeff['a'] * x**2 + coeff['b'] * x + coeff['c']

            return y_eval

    def plot(self, num_points=500, show_points=True, show_legend=True):
        """Vẽ đồ thị hàm spline"""
        plt.figure(figsize=(12, 7))

        x_plot = np.linspace(self.x[0], self.x[-1], num_points)
        y_plot = self.evaluate(x_plot)

        plt.plot(x_plot, y_plot, 'b-', linewidth=2, label=self.spline_type)

        if show_points:
            plt.plot(self.x, self.y, 'ro', markersize=8, label='Du lieu goc')

        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title(f'Do thi ham ghep tron: {self.spline_type}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if show_legend:
            plt.legend(fontsize=10)

        plt.tight_layout()
        return plt

    def print_coefficients(self):
        """In ra các hệ số của hàm spline"""
        print("\n" + "="*80)
        print(f"HE SO CUA HAM SPLINE: {self.spline_type}")
        print("="*80)

        for i, coeff in enumerate(self.coefficients):
            print(f"\nDoan {i+1}: {coeff['interval']}")
            print(f"  Cong thuc: {coeff['formula']}")

            if 'a' in coeff:
                print(f"  a = {coeff['a']}")
            if 'b' in coeff:
                print(f"  b = {coeff['b']}")
            if 'c' in coeff:
                print(f"  c = {coeff['c']}")
            if 'd' in coeff:
                print(f"  d = {coeff['d']}")

        print("\n" + "="*80)


def read_csv_file(filename):
    """Đọc file CSV"""
    print(f"\nDang doc file CSV: {filename}")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

            if ',' in first_line:
                delimiter = ','
            elif ';' in first_line:
                delimiter = ';'
            elif '\t' in first_line:
                delimiter = '\t'
            else:
                delimiter = ','

            print(f"Dau phan cach: '{delimiter}'")

        data = np.genfromtxt(filename, delimiter=delimiter, names=True, encoding='utf-8')

        headers = data.dtype.names
        print(f"\nCac cot trong file: {list(headers)}")

        x_col = input("Nhap ten cot cho x: ").strip()
        y_col = input("Nhap ten cot cho y: ").strip()

        if x_col not in headers or y_col not in headers:
            print("Ten cot khong hop le!")
            return None, None

        x_data = data[x_col].tolist()
        y_data = data[y_col].tolist()

        valid_data = [(x, y) for x, y in zip(x_data, y_data) if not (np.isnan(x) or np.isnan(y))]

        if len(valid_data) < 2:
            print("Du lieu khong du!")
            return None, None

        x_data, y_data = zip(*valid_data)

        return list(x_data), list(y_data)

    except Exception as e:
        print(f"Loi khi doc file CSV: {e}")
        return None, None


def read_xlsx_file(filename):
    """Đọc file XLSX"""
    print(f"\nDang doc file XLSX: {filename}")

    try:
        import openpyxl

        wb = openpyxl.load_workbook(filename)
        sheet = wb.active

        headers = []
        for cell in sheet[1]:
            headers.append(cell.value)

        print(f"\nCac cot trong file: {headers}")

        x_col = input("Nhap ten cot cho x: ").strip()
        y_col = input("Nhap ten cot cho y: ").strip()

        if x_col not in headers or y_col not in headers:
            print("Ten cot khong hop le!")
            return None, None

        x_idx = headers.index(x_col)
        y_idx = headers.index(y_col)

        x_data = []
        y_data = []

        for row in sheet.iter_rows(min_row=2, values_only=True):
            if row[x_idx] is not None and row[y_idx] is not None:
                try:
                    x_data.append(float(row[x_idx]))
                    y_data.append(float(row[y_idx]))
                except (ValueError, TypeError):
                    continue

        return x_data, y_data

    except ImportError:
        print("\nKhong the import openpyxl. Dang cai dat...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
        print("Da cai dat xong! Vui long chay lai chuong trinh.")
        return None, None
    except Exception as e:
        print(f"Loi khi doc file XLSX: {e}")
        return None, None


def input_data_manual():
    """Nhập dữ liệu thủ công"""
    print("\n" + "="*80)
    print("NHAP DU LIEU THU CONG")
    print("="*80)

    while True:
        try:
            n = int(input("\nNhap so luong diem du lieu (toi thieu 2): "))
            if n < 2:
                print("Can it nhat 2 diem! Vui long nhap lai.")
                continue
            break
        except ValueError:
            print("Vui long nhap mot so nguyen hop le!")

    x_data = []
    y_data = []

    print("\nNhap cac cap toa do (x, y):")
    for i in range(n):
        while True:
            try:
                x = float(input(f"  Diem {i+1} - x: "))
                y = float(input(f"  Diem {i+1} - y: "))
                x_data.append(x)
                y_data.append(y)
                break
            except ValueError:
                print("Vui long nhap so hop le!")

    return x_data, y_data


def input_data_file():
    """Đọc dữ liệu từ file"""
    print("\n" + "="*80)
    print("NHAP DU LIEU TU FILE")
    print("="*80)
    print("\nHo tro cac dinh dang:")
    print("  - CSV  (.csv)")
    print("  - Excel (.xlsx)")

    while True:
        file_path = input("\nNhap duong dan file: ").strip()
        file_path = file_path.strip('"').strip("'")

        if not os.path.exists(file_path):
            print("File khong ton tai! Vui long kiem tra lai duong dan.")
            retry = input("Ban co muon thu lai? (y/n): ").lower()
            if retry != 'y':
                return None, None
            continue

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        print(f"\nDinh dang file: {ext}")

        if ext == '.csv':
            x_data, y_data = read_csv_file(file_path)
        elif ext in ['.xlsx', '.xls']:
            x_data, y_data = read_xlsx_file(file_path)
        else:
            print(f"Dinh dang file '{ext}' khong duoc ho tro!")
            retry = input("Ban co muon thu lai? (y/n): ").lower()
            if retry != 'y':
                return None, None
            continue

        if x_data is None or len(x_data) < 2:
            print("Du lieu khong du!")
            retry = input("Ban co muon thu lai? (y/n): ").lower()
            if retry != 'y':
                return None, None
            continue

        print(f"\n✅ Da doc {len(x_data)} diem du lieu tu file!")

        print("\n5 diem dau tien:")
        for i in range(min(5, len(x_data))):
            print(f"  ({x_data[i]}, {y_data[i]})")

        if len(x_data) > 5:
            print(f"  ... va {len(x_data) - 5} diem khac")

        return x_data, y_data


def main():
    """Hàm chính của chương trình"""
    print("="*80)
    print(" "*10 + "CHUONG TRINH GIAI BAI TOAN HAM GHEP TRON - ULTIMATE VERSION")
    print("="*80)

    # Bước 1: Chọn phương thức nhập dữ liệu
    print("\nBUOC 1: CHON PHUONG THUC NHAP DU LIEU")
    print("1. Nhap thu cong")
    print("2. Doc tu file (CSV hoac XLSX)")

    while True:
        choice = input("\nLua chon cua ban (1 hoac 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Lua chon khong hop le!")

    if choice == '1':
        x_data, y_data = input_data_manual()
    else:
        x_data, y_data = input_data_file()
        if x_data is None:
            print("Khong the doc du lieu. Chuong trinh ket thuc.")
            return

    # Bước 2: Chọn số chữ số thập phân
    print("\n" + "="*80)
    print("BUOC 2: CHON SO CHU SO THAP PHAN")
    print("="*80)

    while True:
        try:
            decimal_places = int(input("\nNhap so chu so thap phan sau dau phay (0-10): "))
            if 0 <= decimal_places <= 10:
                break
            print("Vui long nhap so tu 0 den 10!")
        except ValueError:
            print("Vui long nhap so nguyen hop le!")

    # Tạo solver
    solver = SplineSolverUltimate(x_data, y_data, decimal_places)

    print(f"\nDa nhap {solver.n} diem du lieu:")
    for i in range(min(5, solver.n)):
        print(f"  ({solver.x[i]:.{decimal_places}f}, {solver.y[i]:.{decimal_places}f})")
    if solver.n > 5:
        print(f"  ... va {solver.n - 5} diem khac")

    # Bước 3: Chọn loại Spline
    print("\n" + "="*80)
    print("BUOC 3: CHON LOAI SPLINE")
    print("="*80)
    print("1. Spline tuyen tinh (cap 1)")
    print("2. Spline bac 2 voi dieu kien bien S'(x0)")
    print("3. Spline bac 3 tu nhien (Natural: S''(x0)=S''(xn)=0)")
    print("4. Spline bac 3 voi dieu kien bien S'(x0) va S'(xn)")
    print("5. Spline bac 3 voi dieu kien bien S''(x0) va S''(xn)  ⭐ MOI!")
    print("6. Spline bac 3 voi dieu kien bien hon hop  ⭐ MOI!")

    while True:
        spline_choice = input("\nLua chon cua ban (1-6): ").strip()
        if spline_choice in ['1', '2', '3', '4', '5', '6']:
            break
        print("Lua chon khong hop le!")

    # Giải bài toán
    print("\n" + "="*80)
    print("DANG GIAI BAI TOAN...")
    print("="*80)

    if spline_choice == '1':
        solver.linear_spline()

    elif spline_choice == '2':
        print("\nDieu kien bien cho Spline bac 2:")
        print(f"Nhap gia tri S'(x_0) = S'({solver.x[0]})")
        while True:
            try:
                m0 = float(input(f"S'({solver.x[0]}) = "))
                break
            except ValueError:
                print("Vui long nhap so hop le!")

        solver.quadratic_spline(m0)

    elif spline_choice == '3':
        solver.cubic_spline_natural()

    elif spline_choice == '4':
        print("\nDieu kien bien cho Clamped Cubic Spline:")
        print("Nhap gia tri dao ham cap 1 tai hai diem dau va cuoi")

        while True:
            try:
                fp0 = float(input(f"S'({solver.x[0]}) = "))
                fpn = float(input(f"S'({solver.x[-1]}) = "))
                break
            except ValueError:
                print("Vui long nhap so hop le!")

        solver.cubic_spline_clamped(fp0, fpn)

    elif spline_choice == '5':
        print("\n⭐ DIEU KIEN BIEN DAO HAM CAP 2 ⭐")
        print("Nhap gia tri dao ham cap 2 tai hai diem dau va cuoi")

        while True:
            try:
                fpp0 = float(input(f"S''({solver.x[0]}) = "))
                fppn = float(input(f"S''({solver.x[-1]}) = "))
                break
            except ValueError:
                print("Vui long nhap so hop le!")

        solver.cubic_spline_second_derivative(fpp0, fppn)

    else:  # spline_choice == '6'
        print("\n⭐ DIEU KIEN BIEN HON HOP ⭐")
        print("\nDieu kien bien BEN TRAI (x0 = {})".format(solver.x[0]))
        print("1. S'(x0) - Dao ham cap 1")
        print("2. S''(x0) - Dao ham cap 2")

        while True:
            bc_left_type = input("Chon loai dieu kien bien trai (1 hoac 2): ").strip()
            if bc_left_type in ['1', '2']:
                bc_left_type = int(bc_left_type)
                break
            print("Lua chon khong hop le!")

        while True:
            try:
                if bc_left_type == 1:
                    bc_left_value = float(input(f"S'({solver.x[0]}) = "))
                else:
                    bc_left_value = float(input(f"S''({solver.x[0]}) = "))
                break
            except ValueError:
                print("Vui long nhap so hop le!")

        print("\nDieu kien bien BEN PHAI (xn = {})".format(solver.x[-1]))
        print("1. S'(xn) - Dao ham cap 1")
        print("2. S''(xn) - Dao ham cap 2")

        while True:
            bc_right_type = input("Chon loai dieu kien bien phai (1 hoac 2): ").strip()
            if bc_right_type in ['1', '2']:
                bc_right_type = int(bc_right_type)
                break
            print("Lua chon khong hop le!")

        while True:
            try:
                if bc_right_type == 1:
                    bc_right_value = float(input(f"S'({solver.x[-1]}) = "))
                else:
                    bc_right_value = float(input(f"S''({solver.x[-1]}) = "))
                break
            except ValueError:
                print("Vui long nhap so hop le!")

        solver.cubic_spline_mixed(bc_left_type, bc_left_value, bc_right_type, bc_right_value)

    # In kết quả
    solver.print_coefficients()

    # Vẽ đồ thị
    print("\n" + "="*80)
    print("VE DO THI")
    print("="*80)

    plot_choice = input("\nBan co muon ve do thi? (y/n): ").lower()
    if plot_choice == 'y':
        plt_obj = solver.plot()

        save_choice = input("\nBan co muon luu do thi? (y/n): ").lower()
        if save_choice == 'y':
            filename = input("Nhap ten file (khong can duoi .png): ").strip()
            if not filename:
                filename = "spline_plot"

            if not os.path.exists('outputs'):
                os.makedirs('outputs')

            filepath = f"outputs/{filename}.png"
            plt_obj.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"\nDo thi da duoc luu tai: {filepath}")

        plt_obj.show()

    print("\n" + "="*80)
    print("HOAN THANH!")
    print("="*80)


if __name__ == "__main__":
    main()