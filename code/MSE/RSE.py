import numpy as np
import matplotlib.pyplot as plt
import sympy
import sys

def run_least_squares_detailed():
    """
    Thực hiện Phương pháp Bình phương Tối thiểu (Least Squares Method) tổng quát.

    Phiên bản này in chi tiết các ma trận trung gian (Phi, M, b)
    và hệ phương trình tuyến tính Ma = b.
    """

    # Cấu hình numpy in ra đẹp hơn
    np.set_printoptions(precision=4, suppress=True)

    print("--- PHƯƠNG PHÁP BÌNH PHƯƠNG TỐI THIỂU (CHI TIẾT) ---")

    # --- Bước 1: Nhập hàm cơ sở ---
    try:
        m = int(input("Nhập số lượng hàm cơ sở (m): "))
        if m <= 0:
            print("Số lượng hàm cơ sở phải là số nguyên dương.")
            return

        phi_strings = []
        phi_funcs = []
        x_sym = sympy.symbols('x') # Biến symbolic để định nghĩa hàm

        print(f"Nhập {m} hàm cơ sở (sử dụng 'x' làm biến):")
        print("Ví dụ: 1, x, x**2, sin(x), exp(x)")

        for j in range(m):
            while True:
                str_func = input(f"  φ_{j+1}(x) = ")
                try:
                    sym_func = sympy.sympify(str_func)
                    num_func = sympy.lambdify(x_sym, sym_func, 'numpy')

                    phi_strings.append(str_func)
                    phi_funcs.append(num_func)
                    break
                except sympy.SympifyError:
                    print("Lỗi: Không thể phân tích hàm. Vui lòng thử lại.")
                except Exception as e:
                    print(f"Lỗi khi tạo hàm: {e}. Vui lòng thử lại.")

    except ValueError:
        print("Lỗi: Vui lòng nhập một số nguyên.")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return

    # --- Bước 2: Nhập dữ liệu (x_i, y_i) ---
    try:
        n = int(input(f"\nNhập số lượng điểm dữ liệu (n): "))
        if n < m:
            print(f"Lỗi: Số điểm dữ liệu (n={n}) phải lớn hơn hoặc bằng số hàm cơ sở (m={m}).")
            return

        x_data = np.zeros(n)
        y_data = np.zeros(n)

        print(f"Nhập {n} cặp điểm (x, y):")
        for i in range(n):
            while True:
                try:
                    point_str = input(f"  Điểm {i+1} (x y, cách nhau bằng dấu cách): ")
                    x_i, y_i = map(float, point_str.split())
                    x_data[i] = x_i
                    y_data[i] = y_i
                    break
                except ValueError:
                    print("Lỗi: Vui lòng nhập hai số, cách nhau bằng dấu cách.")

        y_vec = y_data.reshape(-1, 1)

    except ValueError:
        print("Lỗi: Vui lòng nhập một số nguyên.")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return

    # --- Bước 3: Xây dựng ma trận Phi (Φ) ---
    Phi = np.zeros((n, m))
    for j in range(m):
        Phi[:, j] = phi_funcs[j](x_data)

    print("\n--- BƯỚC 3: XÂY DỰNG MA TRẬN PHI (Φ) ---")
    print(f"Ma trận Φ (kích thước {n}x{m}), với Φ[i, j] = φ_j(x_i):")
    print(Phi)

    # --- Bước 4: Xây dựng hệ phương trình tuyến tính ---
    try:
        print("\n--- BƯỚC 4: XÂY DỰNG HỆ PHƯƠNG TRÌNH (Ma = b) ---")

        # M = Phi^T * Phi (m x m)
        M = Phi.T @ Phi
        # b = Phi^T * y (m x 1)
        b_vec = Phi.T @ y_vec

        print(f"Ma trận M = Φ^T * Φ (kích thước {m}x{m}):")
        print(M)

        print(f"\nVector b = Φ^T * y (kích thước {m}x1):")
        print(b_vec)

        print("\nHệ phương trình tuyến tính (Ma = b) cần giải (a_j là hệ số của φ_j):")
        for i in range(m): # m là số hàng của M
            equation_str = "  "
            for j in range(m): # m là số cột của M
                equation_str += f"({M[i, j]:.4f}) * a_{j+1}"
                if j < m - 1:
                    equation_str += " + "
            equation_str += f" = {b_vec[i, 0]:.4f}"
            print(equation_str)

        # --- Bước 5: Giải hệ và Tính sai số ---
        print("\n--- BƯỚC 5: GIẢI HỆ VÀ TÍNH SAI SỐ ---")
        a_vec = np.linalg.solve(M, b_vec)

        g_pred = Phi @ a_vec
        S = np.sum((y_vec - g_pred)**2)
        mean_sq_error = np.sqrt(S / n)

        print("Đã giải hệ phương trình, tìm được vector hệ số 'a'.")
        print(f"Tổng bình phương sai số (S): {S:.6f}")

    except np.linalg.LinAlgError:
        print("\n--- LỖI ---")
        print("Không thể giải hệ phương trình. (Ma trận M suy biến)")
        return
    except Exception as e:
        print(f"\nĐã xảy ra lỗi khi giải hệ phương trình: {e}")
        return

    # --- Bước 6: In kết quả ---
    print("\n--- BƯỚC 6: KẾT QUẢ CUỐI CÙNG ---")

    g_x_str = "g(x) = "
    for j in range(m):
        g_x_str += f"({a_vec[j, 0]:.6f}) * ({phi_strings[j]})"
        if j < m - 1:
            g_x_str += " + "

    print(f"Hàm xấp xỉ tìm được:")
    print(g_x_str)

    print("\nCác hệ số a:")
    for j in range(m):
        print(f"  a_{j+1} (cho hàm '{phi_strings[j]}'): {a_vec[j, 0]:.6f}")

    print(f"\nSai số trung bình phương (σ): {mean_sq_error:.6f}")

    # --- Bước 7: Vẽ đồ thị ---
    print("\n--- BƯỚC 7: VẼ ĐỒ THỊ ---")
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, label='Dữ liệu gốc (x_i, y_i)', color='blue', zorder=5)

        x_min, x_max = np.min(x_data), np.max(x_data)
        padding = (x_max - x_min) * 0.05
        x_plot = np.linspace(x_min - padding, x_max + padding, 400)

        y_plot = np.zeros_like(x_plot)
        for j in range(m):
            y_plot += a_vec[j, 0] * phi_funcs[j](x_plot)

        plt.plot(x_plot, y_plot, label=f'Hàm xấp xỉ g(x)', color='red', linewidth=2)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Phương pháp Bình phương Tối thiểu (Chi tiết)')
        plt.legend()
        plt.grid(True)

        plot_filename = 'least_squares_plot_detailed.png'
        plt.savefig(plot_filename)

        print(f"Đã lưu biểu đồ vào file: {plot_filename}")
        print("Hiển thị biểu đồ...")
        plt.show()

    except Exception as e:
        print(f"Đã xảy ra lỗi khi vẽ đồ thị: {e}")

if __name__ == "__main__":
    print("Yêu cầu các thư viện: numpy, matplotlib, sympy")
    print("Bạn có thể cài đặt bằng: pip install numpy matplotlib sympy\n")
    run_least_squares_detailed()