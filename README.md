# 🎓 Citation Network PageRank System 🎓

Hệ thống phân tích mạng trích dẫn học thuật sử dụng thuật toán PageRank và các thuật toán ranking khác để xác định các bài báo và tác giả có ảnh hưởng nhất dựa trên mô hình trích dẫn.

🚀 Live Demo [Here](https://citation-network-frontend.onrender.com) (Free host nên resources hạn chế, tốt hơn nên chạy local, thanks :D)
## 🌟 Tính năng chính 🌟

### 1. **Phân tích theo Tác giả (Author-Based Analysis)**
- Nhập danh sách tên tác giả
- Tự động thu thập các bài báo của tác giả từ Semantic Scholar API
- Xây dựng mạng trích dẫn và tính toán điểm PageRank
- Xác định các bài báo có ảnh hưởng nhất trong lĩnh vực nghiên cứu

### 2. **Phân tích theo Bài báo (Paper-Based Analysis)**
- Hỗ trợ nhiều định dạng đầu vào: Title, DOI, ArXiv ID, PubMed ID
- Phân tích mối quan hệ trích dẫn giữa các bài báo cụ thể
- Xếp hạng bài báo theo độ quan trọng

### 3. **Nhiều Thuật toán Ranking**
- **PageRank**: Thuật toán gốc của Google để xếp hạng trang web
- **Weighted PageRank**: Phiên bản PageRank có trọng số dựa trên số lần trích dẫn
- **HITS (Hyperlink-Induced Topic Search)**: Tính toán Hub và Authority scores

### 4. **So sánh Thuật toán (Multi-Algorithm Comparison)**
- Chạy và so sánh nhiều thuật toán cùng lúc
- Tính toán Spearman Rank Correlation
- Phân tích Top-K Overlap
- So sánh Performance metrics và Convergence curves

### 5. **Trực quan hóa Mạng (Interactive Graph Visualization)**
- Biểu đồ mạng tương tác 2D với D3.js
- Hiển thị nodes (bài báo) và edges (trích dẫn)
- Zoom, pan và tương tác với từng node
- Màu sắc và kích thước node phản ánh độ quan trọng

### 6. **Network Metrics**
- Density, Average Degree, Clustering Coefficient
- Hub và Authority identification
- Degree Distribution analysis
- Strongly connected nodes, Dangling nodes

### 7. **Convergence Analysis**
- Convergence curves cho từng thuật toán
- Theo dõi quá trình hội tụ qua các iterations
- Residual tracking

### 8. **Role-Based Access Control**
- **Researcher**: Chức năng cơ bản (ranking results, basic visualization)
- **Data Scientist**: Full access với performance comparison, network metrics và convergence analysis

## 🛠️ Công nghệ sử dụng

### Backend
- **Flask**: Web framework cho Python
- **Semantic Scholar API**: Thu thập dữ liệu bài báo học thuật
- **NetworkX**: Xử lý và phân tích đồ thị mạng
- **NumPy**: Tính toán số học và đại số tuyến tính
- **SciPy**: Tính toán khoa học

### Frontend
- **React**: UI framework
- **D3.js**: Trực quan hóa đồ thị tương tác
- **Chart.js**: Vẽ biểu đồ convergence và performance
- **React-Force-Graph-2D**: Render mạng trích dẫn
- **React-Markdown**: Hiển thị nội dung hướng dẫn

## Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **Node.js**: 14.x trở lên
- **npm**: 6.x trở lên
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB cho mạng lớn)
- **Kết nối Internet**: Cần thiết để truy cập Semantic Scholar API

## Data Source

This system uses **Semantic Scholar API** which provides:
- 200M+ academic papers
- Citation relationships
- Author information
- Publication metadata

## ⚠️ LƯU Ý QUAN TRỌNG VỀ API KEY

### Giới hạn khi sử dụng API Key mặc định

Hệ thống hiện tại khi bạn Rerun, chỉ sử dụng **Semantic Scholar API không có API key đăng ký**. Điều này có nghĩa là bạn sẽ gặp phải các giới hạn sau:

- **100 requests/5 phút** cho public API
- Nếu vượt quá giới hạn, bạn sẽ nhận được lỗi `429 Too Many Requests`
- Hệ thống sẽ tự động chờ và thử lại, nhưng quá trình xử lý sẽ chậm hơn

Để có trải nghiệm tốt hơn:
### Thực hiện các bước dưới đây:
1. **Đăng ký API key miễn phí** tại: https://www.semanticscholar.org/product/api#api-key-form
   - Với API key: **5000 requests/5 phút**
   - Tốc độ xử lý nhanh hơn và ổn định hơn

2. **Cấu hình API key** trong code:
   
   Mở file `app.py` và thêm API key của bạn:
   
   ```python
   # Tìm dòng này trong app.py
   API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY')  # Default
   
   # Thay đổi thành
   API_KEY = = "YOUR_API_KEY_HERE"
   ```

3. **Hoặc sử dụng biến môi trường** (khuyến nghị):
   
   ```bash
   # Windows
   set SEMANTIC_SCHOLAR_API_KEY=your_api_key_here
   python app.py
   
   # macOS/Linux
   export SEMANTIC_SCHOLAR_API_KEY=your_api_key_here
   python app.py
   ```
### Đối với các bạn Reviewer cần reproduce source code: Mình có cung cấp API Key của mình trong Báo cáo tại Chương 6, các bạn có thể lấy Key và thực hiện các bước trên nhé. 
### Hoặc Sử dụng anonymous API key được config mặc định trong source.
#### 💡 Tips khi sử dụng API mặc định

- **Giảm số lượng tác giả/bài báo** trong một lần phân tích (2-3 inputs)
- **Tránh chạy nhiều request liên tiếp** trong thời gian ngắn
- **Chờ 5 phút** nếu gặp lỗi rate limit trước khi thử lại
- **Sử dụng cache** - hệ thống đã tự động cache kết quả để giảm số lần gọi API


## 🚀 Hướng dẫn cài đặt và chạy

### Bước 1: Clone Repository

```bash
git clone https://github.com/HungPham2002/citation-network-pagerank-system.git
cd citation-network-pagerank-system
```

### Bước 2: Cài đặt Backend

#### 2.1. Tạo môi trường ảo Python

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 2.2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

#### 2.3. Chạy Flask server

```bash
python app.py
```

Backend sẽ chạy tại: `http://localhost:5001`

### Bước 3: Cài đặt Frontend

#### 3.1. Mở terminal mới và di chuyển vào thư mục frontend

```bash
cd frontend
```

#### 3.2. Cài đặt Node dependencies

```bash
npm install
```

#### 3.3. Chạy React development server

```bash
npm start
```

Frontend sẽ tự động mở tại: `http://localhost:3000`

### Bước 4: Sử dụng ứng dụng

1. **Chọn Role**: Researcher / Data Scientist
2. **Chọn Input Mode**: Authors hoặc Papers
3. **Nhập dữ liệu**:
   - **Authors**: Nhập tên tác giả (mỗi dòng một tên)
     ```
     Tho Quan
     Yoshua Bengio
     Yann LeCun
     ```
   - **Papers**: Nhập title hoặc DOI/ArXiv ID (mỗi dòng một bài)
     ```
     10.1109/CVPR.2016.90 
     arXiv:1706.03762  
     1810.04805
     10.48550/arXiv.2010.11929
     2103.00020
    ```
     Deep Residual Learning for Image Recognition
     Attention Is All You Need
     BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
     An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
     Learning Transferable Visual Models From Natural Language Supervision
4. **Chọn Algorithm (DS)**: Single hoặc Multiple algorithms
5. **Điều chỉnh Parameters (DS)**:
   - Damping Factor: 0.85 (mặc định)
   - Max Iterations: 100 (mặc định)
6. **Click "Calculate"** và xem kết quả

## 📁 Cấu trúc thư mục

```
citation-network-pagerank-system/
├── app.py                      # Flask backend server
├── requirements.txt            # Python dependencies
├── package.json               # Root package.json
├── README.md                  # Tài liệu này
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── App.js            # Main React component
│   │   ├── App.css           # Styles
│   │   └── index.js          # Entry point
│   ├── public/               # Static assets
│   ├── package.json          # Frontend dependencies
│   └── README.md             # Create React App docs
└── arn_venv/                 # Python virtual environment (local)
```

## ✅ Update fix logs (24/11/2025)

- [x] Fix bug không hiển thị Interactive Graph Visualization khi chạy mode single algorithm đối với 2 thuật toán Weighted PageRank và HITS
- [x] Fix bug không hiển thị Interactive Graph Visualization khi chạy mode multi algorithm khi có 1 trong 2 thuật toán đã nêu
- [x] Fix bug giao diện bị overlap khi so sánh Performance metric
- [x] Add Convergence Curve vào output DS role
- [x] Fix bug state không được clear khi change role
- [x] Fix bug So sánh multi algorithms - Paper analyzed không được trả về
- [x] Fix bug Convergence Curve không được backend trả về đúng cách
- [x] Fix bug hiển thị cho Convergence Curve

## TODO Fix logs 
- [ ] Bổ sung logic kiểm tra cross-reference giữa các input papers.
- [ ] Fix bug thanh tiến trình không hoạt động khi Run so sánh Multi Algorithms (Don't worry, be patient. Backend still working 'til the end).

## Troubleshooting

### Lỗi: "Semantic Scholar API not available"
- Kiểm tra kết nối internet
- API có thể bị rate limit, đợi vài phút và thử lại

### Lỗi: Port đã được sử dụng
- Backend: Thay đổi PORT trong app.py
- Frontend: Sử dụng `PORT=3001 npm start`

### Lỗi: Module not found
```bash
# Backend
pip install -r requirements.txt --force-reinstall

# Frontend
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Lỗi CORS
- Kiểm tra flask-cors đã được cài đặt
- Đảm bảo backend chạy trước frontend

## Use Cases

- **Researchers**: Tìm các bài báo có ảnh hưởng trong lĩnh vực nghiên cứu
- **Data Scientists**: Phân tích patterns và trends trong citation networks
- **Academic Institutions**: Đánh giá research impact và ranking
- **Students**: Khám phá các bài báo nền tảng trong lĩnh vực học tập


## Tác giả

- **Phạm Hữu Hùng** — Postgraduate Student (ID: 2470299) • [CV (PDF)](https://github.com/HungPham2002/resume/blob/main/Resume_HungPham.pdf)
- **Võ Thị Vân Anh** — Postgraduate Student (ID: 2470283)

## Acknowledgments
- Tác giả xin chân thành cảm ơn CN. Lê Nho Hãn và CN. Vũ Trần Thanh Hương đã có những góp ý quý báu và những nhận xét sâu sắc trong suốt quá trình nghiên cứu và thực hiện đồ án.
- [Semantic Scholar API](https://www.semanticscholar.org/product/api) - Cung cấp dữ liệu bài báo học thuật
- [PageRank Algorithm](https://en.wikipedia.org/wiki/PageRank) - Larry Page & Sergey Brin
- [HITS Algorithm](https://en.wikipedia.org/wiki/HITS_algorithm) - Jon Kleinberg

## Contact
Email: phhung.sdh241@hcmut.edu.vn • vtvanh.sdh241@hcmut.edu.vn
---
