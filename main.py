import cv2
import numpy as np

def show_scaled_window(win_name, img, max_height=800):
    """按比例缩放图像以适应屏幕显示"""
    if img is None: return
    height, width = img.shape[:2]
    scale = max_height / height
    new_width = int(width * scale)
    display_img = cv2.resize(img, (new_width, max_height))
    cv2.imshow(win_name, display_img)

def main():
    # 1. 读取大图和模板
    image_path = 'level_image.jpg'
    template_path = 'template_up.jpg'
    
    img = cv2.imread(image_path)
    # 直接在灰度图上找，不需要二值化了，保留更多细节
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    template_up = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"错误: 找不到大图 '{image_path}'。")
        return
    if template_up is None:
        print(f"错误: 找不到模板文件 '{template_path}'。请确保你截取了一个向上的箭头并正确命名。")
        return

    h, w = template_up.shape
    print(f"成功加载模板，尺寸: {w}x{h}")

    # 2. 生成四个方向的模板
    templates = {
        'N': template_up,
        'S': cv2.rotate(template_up, cv2.ROTATE_180),
        'E': cv2.rotate(template_up, cv2.ROTATE_90_CLOCKWISE),
        'W': cv2.rotate(template_up, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }

    final_output = img.copy() # 在原图的拷贝上绘制，直观对比
    arrows_found = []

    # 3. 匹配阈值 (0 到 1 之间。如果漏识别就调低，如果误识别了杂物就调高)
    threshold = 0.87 

    viz_colors = {'N':(0,0,255), 'S':(255,0,0), 'W':(0,255,0), 'E':(0,255,255)}
    
    # 4. 遍历四个方向进行扫描
    for dir_name, temp in templates.items():
        # 执行模板匹配
        res = cv2.matchTemplate(gray_img, temp, cv2.TM_CCOEFF_NORMED)
        
        # 找到所有匹配度大于阈值的坐标
        loc = np.where(res >= threshold)
        
        # 将坐标打包成 (x, y) 的列表
        points = list(zip(*loc[::-1]))
        
        # --- 核心过滤：去重 (非极大值抑制的简化版) ---
        # matchTemplate 会在箭头周围产生多个相近的匹配点，我们需要把距离太近的点合并
        filtered_points = []
        for pt in points:
            is_duplicate = False
            for f_pt in filtered_points:
                # 如果两个点距离小于模板宽度的一半，认为是同一个箭头
                distance = np.sqrt((pt[0] - f_pt[0])**2 + (pt[1] - f_pt[1])**2)
                if distance < w / 2:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_points.append(pt)
                arrows_found.append({'dir': dir_name, 'coord': pt})
                
                # 绘制结果
                center_x = pt[0] + w//2
                center_y = pt[1] + h//2
                cv2.circle(final_output, (center_x, center_y), max(w//2, 8), viz_colors[dir_name], 2)
                cv2.putText(final_output, dir_name, (center_x - 5, center_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, viz_colors[dir_name], 2)

    print(f"扫描完毕！总共识别出 {len(arrows_found)} 个箭头。")

    show_scaled_window('Template Matching Result', final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()