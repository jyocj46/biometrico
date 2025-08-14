import cv2
import face_recognition
import time

# Configurables
USE_CNN = False           # True = mas preciso pero mas lento (y mejor con GPU)
DOWNSCALE = 0.5           # 0.5 = procesa a media resolucion para mas FPS
CAM_INDEX = 0             # Si no abre la camara, prueba 1, 2, etc.

def main():
    # En Windows a veces ayuda CAP_DSHOW
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara. Prueba otro CAM_INDEX.")

    model = "cnn" if USE_CNN else "hog"
    print("Presiona 'q' para salir")

    prev_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer frame de la camara.")
            break

        # Reduccion para velocidad
        small = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE)
        rgb_small = small[:, :, ::-1]  # BGR -> RGB

        # Deteccion de caras
        boxes_small = face_recognition.face_locations(rgb_small, model=model)

        # Reescalar boxes al tamaño original
        scale = 1.0 / DOWNSCALE
        boxes = [(int(top*scale), int(right*scale), int(bottom*scale), int(left*scale))
                 for (top, right, bottom, left) in boxes_small]

        # Dibujar rectangulos
        for i, (top, right, bottom, left) in enumerate(boxes, 1):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
            cv2.putText(frame, f"Face {i}", (left, max(20, top-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # FPS simple
        now = time.time()
        fps = 1.0 / (now - prev_t) if now > prev_t else 0
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}  model:{model}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Deteccion de rostros (face_recognition)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
