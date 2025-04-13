# Image Inpaiting using Conditional Flow Matching - Exercise M09W05 AIO2024

**Flow Matching** là một phương pháp hiệu quả để huấn luyện các mô hình sinh (generative models) dựa trên dòng (flow). Khác với Diffusion Models, Flow Matching tập trung vào việc học vận tốc chuyển đổi giữa phân phối dữ liệu và phân phối noise, sau đó tạo mẫu bằng cách đi theo vận tốc này.

Trong bài toán mô hình sinh, mục tiêu là tạo ra các mẫu mới từ một phân phối cơ bản (phân phối dữ liệu huấn luyện). Gọi:
- $p_0(x_0)$ là phân phối dữ liệu thực <br>
- $p_1(x_1)$ là phân phối noise (thường là phân phối Gaussian) <br>

Mục tiêu của mô hình sinh là học một hàm ánh xạ ϕ sao cho:
- $X_0 \sim p_0$ (dữ liệu thực) <br>
- $X_1 = \phi(X_0) \sim p_1$ (noise) <br>

Flow Models định nghĩa một quá trình chuyển đổi liên tục giữa hai phân phối. Đối với mỗi thời điểm t ∈ [0,1], ta có một phân phối trung gian pt và một ánh xạ ϕt+h|t sao cho
- $X_t \sim p_t$ <br>
- $X_{t+h} = \varphi_{t+h|t}(X_t)$ <br>

Flow Matching là một phương pháp hiệu quả để huấn luyện các mô hình flow bằng cách:
- Học một trường vận tốc (velocity field) thay vì học trực tiếp hàm ánh xạ <br>
- Lấy mẫu bằng cách theo dõi vận tốc này qua phương trình vi phân <br>

Trong Flow Matching, vận tốc đại diện cho tốc độ chuyển đổi giữa dữ liệu sạch và dữ liệu nhiễu. Cụ thể:
- $v_t$: vận tốc tại thời điểm t <br>
- $\Psi_t(x)$: flow ánh xạ tại thời điểm t <br>
- $u_t(x)$: vận tốc thực, được tính bằng x1 −x0 <br>

**1.1. Huấn luyện mô hình**

![Training Flow Matching](/readme_img/training_FM.png 'AIO2024')

Để cập nhật trọng số mô hình dự đoán θ, quá trình huấn luyện Flow Matching cho mỗi epoch bao gồm các bước sau:
 (a) Chọn ngẫu nhiên một mẫu dữ liệu sạch $x_0 \sim p_0$ <br>
 (b) Chọn ngẫu nhiên một mẫu noise $x_1 \sim p_1$ <br>
 (c) Chọn ngẫu nhiên một thời điểm t ∈ [0,1] <br>
 (d) Tính điểm nội suy (interpolated point) $x_t = (1 - t)x_0 + tx_1$ <br>
 (e) Tính vận tốc thực $u_t = x_1 - x_0$ <br>
 (f) Dự đoán vận tốc bằng mô hình $v_t = f_{\theta}(x_t, t)$ <br>
 (g) Tính hàm mất mát $l_t = \left |v_t - u_t \right |^2$ <br>
 (h) Cập nhật tham số mô hình θ bằng gradient descent <br>

Hàm mất mát tổng quát có thể viết dưới dạng:
$$\pounds = E_{t, X_0, X_1}\left | u_t^\theta(X_t) - (X_1 - X_0)\right |^2$$
Trong đó: <br>
    $X_t = (1 - t)X_0 + tX_1$ <br>
    $X_0 \sim p_0$ <br>
    $X_1 \sim p_1$ <br>
    t ∼ U(0,1) <br>

**1.2. Lấy mẫu**

![Sampling Flow Matching](/readme_img/Sampling_FM.png 'AIO2024')

Sau khi huấn luyện mô hình dự đoán vận tốc fθ(y,t), chúng ta có thể lấy mẫu bằng cách giải phương trình vi phân:
$$\frac{dy}{dt} = f_\theta(y,t)$$
 Phương pháp phổ biến để giải phương trình này là phương pháp Euler:
 y(t +∆t) ≈ y(t)+fθ(y,t)∆t $$y(t + \Delta t) \approx y(t) + f_\theta(y, t)\Delta t$$
 Quá trình lấy mẫu bao gồm các bước: <br>
 (a) Khởi tạo y từ phân phối noise $p_1$ <br>
 (b) Chia khoảng thời gian [0,1] thành n bước nhỏ: $t_1$, với $\Delta t = t_i - t_{i-1}$ <br>
 (c) Tại mỗi bước i: <br>
    &nbsp;&nbsp;&nbsp;&nbsp; - Tính vận tốc $f_\theta(y, t_i)$ <br>
    &nbsp;&nbsp;&nbsp;&nbsp; - Cập nhật $y = t + f_\theta(y, t_i)$ <br>
 (d) Kết quả cuối cùng là mẫu từ phân phối $p_0$