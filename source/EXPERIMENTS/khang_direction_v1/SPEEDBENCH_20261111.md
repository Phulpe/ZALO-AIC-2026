# Speedbench prefix-cache (gd3 kernel v3, 11/11/2026)
- run_id khang_dir_v1_speedbench_prefix_20261111; 4 câu probe no-hint, cùng ID/seed với gd3 v2. Output PRIVATE_DATA/khang_direction_v1/gd3_v3. GPU 5187 s.
- Kết quả: 4912 s (trước 4961 s) — không cải thiện. 37 lượt gọi: prefill 775 s (~21 s/lượt), decode 4031 s cho 28155 token (~7 token/s, ~760 token/lượt). Decode chiếm 84%.
- prefix_hits = 0: snapshot system không kích hoạt (system_prefix_len trả 0 — template không tách được phần system). Không đáng sửa vì prefill chỉ chiếm 16%.
- GPU1 không nhận job (4 câu cùng affinity doanh nghiệp).
- Kết luận: nút thắt là decode bnb 4-bit trên T4. Máy chấm chính thức 2×RTX 5090 + vLLM bf16 sẽ nhanh hơn nhiều lần; không tiêu thêm GPU Kaggle để tối ưu T4. Câu trả lời giống lần trước (3 answered/4).
