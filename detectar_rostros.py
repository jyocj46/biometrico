import cv2
import face_recognition
import time
import numpy as np
import psycopg2
from psycopg2 import Binary

# ================== Config ==================
USE_CNN   = False     # True = mas preciso pero mas lento
DOWNSCALE = 0.5       # 0.5 = mas FPS
CAM_INDEX = 0         # cambia a 1/2 si no abre

PG_CFG = dict(
    host="localhost",
    port=5432,
    dbname="reconocimiento",  # tu BD
    user="admin",          # tu usuario
    password="admin"    # tu password
)
# ============================================

def get_pg_conn():
    return psycopg2.connect(**PG_CFG)

def insertar_usuario(nombre: str, apellido: str, rostro_bgr: np.ndarray) -> int:
    """
    codifica el recorte del rostro a jpg y lo guarda en la tabla 'usuarios'
    columnas: nombre (text), apellido (text), rostro (bytea)
    """
    ok, buf = cv2.imencode(".jpg", rostro_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("no se pudo codificar el recorte a JPEG")
    jpg_bytes = buf.tobytes()

    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO usuarios (nombre, apellido, rostro) VALUES (%s, %s, %s) RETURNING id",
            (nombre, apellido, Binary(jpg_bytes))
        )
        uid = cur.fetchone()[0]
        conn.commit()
        return uid

def main():
    # en windows ayuda CAP_DSHOW
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("no se pudo abrir la camara. prueba otro CAM_INDEX")

    model = "cnn" if USE_CNN else "hog"
    print("controles:  'r' = registrar  |  'q' = salir")

    prev_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("no se pudo leer frame de la camara")
            break

        # reducir para velocidad
        small = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE)
        rgb_small = small[:, :, ::-1]  # BGR -> RGB

        # deteccion de caras
        boxes_small = face_recognition.face_locations(rgb_small, model=model)

        # reescalar boxes al tamano original
        scale = 1.0 / DOWNSCALE
        boxes = [(int(t*scale), int(r*scale), int(b*scale), int(l*scale))
                 for (t, r, b, l) in boxes_small]

        # dibujar rectangulos
        for i, (top, right, bottom, left) in enumerate(boxes, 1):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
            cv2.putText(frame, f"Face {i}", (left, max(20, top-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # fps
        now = time.time()
        fps = 1.0 / (now - prev_t) if now > prev_t else 0
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}  model:{model}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("registro de rostro (solo guardar en BD)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # ===== registrar =====
        if key == ord('r'):
            if not boxes:
                print("no hay rostro claro para registrar. acercate y mira a la camara")
                continue

            # primer rostro detectado en el frame completo
            top, right, bottom, left = boxes[0]
            rostro_bgr = frame[top:bottom, left:right]

            if rostro_bgr.size == 0 or (bottom - top) <= 0 or (right - left) <= 0:
                print("recorte vacio/incorrecto, intenta de nuevo")
                continue

            # convertir a RGB + asegurar contigüidad en memoria (evita el TypeError de dlib)
            rostro_rgb = cv2.cvtColor(rostro_bgr, cv2.COLOR_BGR2RGB)
            rostro_rgb = np.ascontiguousarray(rostro_rgb, dtype=np.uint8)

            # indicar que TODO el recorte es el rostro (evita redetección dentro del recorte)
            h, w = rostro_rgb.shape[:2]
            full_box = [(0, w, h, 0)]
            encs = face_recognition.face_encodings(rostro_rgb, known_face_locations=full_box, num_jitters=1)

            if not encs:
                print("no se pudo validar el rostro (sin encoding). mejora la luz o reencuadra")
                continue

            try:
                nombre = input("Nombre: ").strip() or "sin_nombre"
                apellido = input("Apellido: ").strip() or "sin_apellido"
            except Exception:
                nombre, apellido = "sin_nombre", "sin_apellido"

            try:
                uid = insertar_usuario(nombre, apellido, rostro_bgr)  # seguimos guardando el JPEG del BGR
                print(f"registrado OK -> id={uid}, {nombre} {apellido}")
            except Exception as e:
                print("error al insertar en BD:", e)


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
