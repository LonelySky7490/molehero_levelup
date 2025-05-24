import cv2
import numpy as np
import pyautogui
import time
from ultralytics import YOLO
from PIL import ImageGrab

# 加载YOLO模型
model = YOLO('best.pt')

# 设置目标类别
target_class = ['FlowerEater', 'ToyMan', 'yes']
confidence_threshold = 0.5  # 可以根据需要调整这个值


def capture_screen():
    # 捕获屏幕
    screen = np.array(ImageGrab.grab())
    # 转换颜色空间从BGR到RGB
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    return screen


def click_target(x, y):
    # 移动鼠标并点击
    pyautogui.moveTo(x, y)
    pyautogui.click()
    print(f"已点击位置 ({x}, {y})")


def main():
    print("开始屏幕检测...")
    last_click_time = 0  # 记录上次点击时间
    try:
        while True:
            current_time = time.time()
            # 捕获屏幕
            screen = capture_screen()

            # 使用YOLO模型进行检测
            results = model(screen)

            # 处理检测结果
            targets = []  # 存储检测到的目标
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 获取置信度
                    confidence = float(box.conf[0])

                    # 如果置信度低于阈值，跳过这个检测结果
                    if confidence < confidence_threshold:
                        continue

                    # 获取类别
                    cls = int(box.cls[0])
                    class_name = result.names[cls]

                    # 如果检测到目标类别
                    if class_name in target_class:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # 计算中心点
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        targets.append((center_x, center_y, class_name))

                        # 在屏幕上绘制边界框
                        cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(screen, class_name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # 绘制中心点（红色圆点）
                        cv2.circle(screen, (center_x, center_y), 5, (0, 0, 255), -1)


            # 显示检测结果
            # 在显示检测结果前添加（放在cv2.imshow()之前）
            display_size = (960, 540)  # 设置你想要显示的窗口大小
            resized_screen = cv2.resize(screen, display_size)
            cv2.imshow('Screen Detection', resized_screen)

            # 每隔5秒点击检测到的目标
            if targets and current_time - last_click_time >= 5:
                # 只点击第一个检测到的目标（可根据需求修改）
                target = targets[0]
                click_target(target[0], target[1])
                last_click_time = current_time
                print(f"检测到 {target[2]}，等待下一次点击...")

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 短暂延迟以减少CPU负载
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n程序已停止")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()