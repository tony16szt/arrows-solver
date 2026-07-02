import cv2
import numpy as np

def show_scaled_window(win_name, img, max_height=800):
    if img is None: return
    height, width = img.shape[:2]
    scale = max_height / height
    new_width = int(width * scale)
    display_img = cv2.resize(img, (new_width, max_height))
    cv2.imshow(win_name, display_img)

def main():
    image_path = 'level_image.jpg'
    template_path = 'template_up.jpg'
    
    img = cv2.imread(image_path)
    template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if img is None or template_gray is None:
        print("错误: 找不到图片或模板文件。")
        return

    # ====== 核心修复 1：反向二值化 ======
    # 将原图和模板都变成“黑底白字”的纯粹二值图。彻底抹除边缘抗锯齿的干扰！
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV)
    _, binary_template = cv2.threshold(template_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # ====== 核心修复 2：加回 UI 遮罩 ======
    # 把顶部和底部的非游戏区域在二值图上涂成纯黑，防止匹配到 UI 按钮
    h, w = binary_img.shape
    cv2.rectangle(binary_img, (0, 0), (w, int(h * 0.12)), 0, -1) 
    cv2.rectangle(binary_img, (0, int(h * 0.88)), (w, h), 0, -1) 

    t_h, t_w = binary_template.shape

    # 2. 生成四个方向的纯白模板
    templates = {
        'N': binary_template,
        'S': cv2.rotate(binary_template, cv2.ROTATE_180),
        'E': cv2.rotate(binary_template, cv2.ROTATE_90_CLOCKWISE),
        'W': cv2.rotate(binary_template, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }

    final_output = img.copy() 
    arrows_found = []

    # ====== 核心修复 3：降低阈值 ======
    # 因为变成了纯黑白匹配，0.75 已经是一个非常安全且高召回率的值
    threshold = 0.3 

    viz_colors = {'N':(0,0,255), 'S':(255,0,0), 'W':(0,255,0), 'E':(0,255,255)}
    
    # 3. 遍历四个方向进行扫描
    for dir_name, temp in templates.items():
        # 注意：这里改用 binary_img 进行匹配
        res = cv2.matchTemplate(binary_img, temp, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        points = list(zip(*loc[::-1]))
        
        filtered_points = []
        for pt in points:
            is_duplicate = False
            for f_pt in filtered_points:
                dist = np.sqrt((pt[0] - f_pt[0])**2 + (pt[1] - f_pt[1])**2)
                if dist < t_w / 2:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_points.append(pt)
                arrows_found.append({'dir': dir_name, 'coord': pt})
                
                center_x = pt[0] + t_w//2
                center_y = pt[1] + t_h//2
                cv2.circle(final_output, (center_x, center_y), max(t_w//2, 8), viz_colors[dir_name], 2)
                cv2.putText(final_output, dir_name, (center_x - 5, center_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, viz_colors[dir_name], 2)

    print(f"扫描完毕！总共识别出 {len(arrows_found)} 个箭头。")

    # 为了让你清楚看到计算机的“视野”，我把去除了UI的二值图也显示出来
    show_scaled_window('Debug: Binary View', binary_img)
    show_scaled_window('Template Matching Result V2', final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()