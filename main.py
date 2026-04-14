#pip install ultralytics opencv-python
import cv2
from ultralytics import YOLO

# Carrega o modelo YOLOv8
model = YOLO("yolov8n.pt")

# Inicializa a câmera (0 = padrão)
cap = cv2.VideoCapture(0)

cv2.namedWindow("YOLOv8", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção de objetos no frame
    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # Desenha a caixa e o rótulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} ({conf:.2f})"
            cv2.putText(frame, text, (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Object Detection - YOLOv8", frame)

    # Pressione ESC para sair
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

