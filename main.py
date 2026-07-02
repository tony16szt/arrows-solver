import cv2
import numpy as np

thresh = None
eroded = None
current_k = 3 # 记录当前滑块的值

def on_trackbar(val):
    global thresh, eroded, current_k
    current_k = val if val > 0 else 1
    kernel = np.ones((current_k, current_k), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    
    h, w = eroded.shape[:2]
    scale = 800 / h
    display_img = cv2.resize(eroded, (int(w * scale), 800))
    cv2.imshow('1. Adjust Slider to Melt Lines', display_img)

def main():
    global thresh, eroded, current_k
    image_path = 'level_image.jpg'
    img = cv2.imread(image_path)
    if img is None:
        print("错误: 找不到图片。")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape
    cv2.rectangle(thresh, (0, 0), (w, int(h * 0.12)), 0, -1) 
    cv2.rectangle(thresh, (0, int(h * 0.88)), (w, h), 0, -1) 

    cv2.namedWindow('1. Adjust Slider to Melt Lines', cv2.WINDOW_AUTOSIZE)
    
    print("\n" + "="*50)
    print("【终极操作指南】")
    print("1. 往右拖动滑块，建议直接拖到 6。")
    print("2. 确认细线消失只剩光点后，按下键盘上的 回车键 (Enter)！")
    print("="*50 + "\n")

    cv2.createTrackbar('Kernel Size', '1. Adjust Slider to Melt Lines', 3, 20, on_trackbar)
    on_trackbar(6) # 默认直接给 6

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13: 
            break
    
    cv2.destroyWindow('1. Adjust Slider to Melt Lines')

    # === 终极逻辑：隔离审讯与拐角排雷 ===
    
    # 1. 找出所有原始的完整连通线条（没融化之前的）
    original_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_output = img.copy()
    arrows_found = 0

    # 2. 遍历每一根线条，单独拉进“小黑屋”处理
    for cnt in original_contours:
        if cv2.contourArea(cnt) < 100: continue # 过滤微小噪点

        # 建立一个小黑屋（空白遮罩），并把这根线画进去
        isolated_mask = np.zeros_like(thresh)
        cv2.drawContours(isolated_mask, [cnt], 0, 255, thickness=-1)

        # 在小黑屋里，只对这根线使用你刚才选好的参数进行腐蚀
        kernel = np.ones((current_k, current_k), np.uint8)
        isolated_eroded = cv2.erode(isolated_mask, kernel, iterations=1)

        # 寻找腐蚀后留下来的小光点
        dot_contours, _ = cv2.findContours(isolated_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not dot_contours: continue # 如果这根线全被融化了（说明它没有箭头），跳过

        # 找到最大的残留点（通常就是箭头核心）
        dot_cnt = max(dot_contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(dot_cnt)
        if max(bw, bh) < 3: continue # 过滤掉极微小的灰尘

        cx = x + bw // 2
        cy = y + bh // 2

        # --- 开始拐角排雷与方向侦测 ---
        # 探测半径：稍微超出光点一点点
        offset = int(max(bw, bh) * 0.8) + 3 
        win = 4 # 探测窗口大小
        
        # 在【隔离的小黑屋】里，探测东南西北四个方向有没有这根线的身体
        # 因为在小黑屋里，绝对不会碰到别人的身体！
        regions = {
            'N': isolated_mask[max(0, cy-offset-win):max(0, cy-offset+win), max(0, cx-win):min(w, cx+win)],
            'S': isolated_mask[max(0, cy+offset-win):min(h, cy+offset+win), max(0, cx-win):min(w, cx+win)],
            'W': isolated_mask[max(0, cy-win):min(h, cy+win), max(0, cx-offset-win):max(0, cx-offset+win)],
            'E': isolated_mask[max(0, cy-win):min(h, cy+win), max(0, cx+offset-win):min(w, cx+offset+win)]
        }

        # 计算四个方向的白色像素数量
        scores = {d: np.sum(r) / 255.0 for d, r in regions.items()}
        
        # 认为“有尾巴”的阈值：只要有超过 5 个白色像素，就说明那边连着线
        active_tails = [d for d, score in scores.items() if score > 5]

        # 【核心排雷逻辑】：
        # 如果有 2 条尾巴，说明这是一个没融化干净的 90 度拐角！直接毙掉！
        # 如果只有 1 条尾巴，那才是真正的箭头！
        if len(active_tails) != 1:
            continue

        tail_dir = active_tails[0]
        dir_map = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
        arrow_dir = dir_map[tail_dir] # 箭头的方向和尾巴相反
        
        arrows_found += 1
        
        # 绘制结果
        viz_colors = {'N':(0,0,255), 'S':(255,0,0), 'W':(0,255,0), 'E':(0,255,255)}
        cv2.circle(final_output, (cx, cy), 8, viz_colors[arrow_dir], 2)
        cv2.putText(final_output, arrow_dir, (cx - 10, cy - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, viz_colors[arrow_dir], 2)

    print(f"提取完成！精准识别出 {arrows_found} 个箭头。")

    scale = 800 / h
    new_width = int(w * scale)
    display_img = cv2.resize(final_output, (new_width, 800))
    cv2.imshow('Final Result (Isolated & Filtered)', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()