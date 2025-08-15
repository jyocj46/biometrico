# login_rt.py
import cv2
import face_recognition
import numpy as np
import psycopg2

# ===== Config =====
USE_CNN   = False     # usa 'hog' por defecto; True cambia a 'cnn' (mas preciso, mas lento)
DOWNSCALE = 0.5       # factor de reduccion para mas FPS (0.5 recomendado)
CAM_INDEX = 0         # cambia si tu camara no abre

PG_CFG = dict(
    host="localhost",
    port=5432,
    dbname="reconocimiento",
    user="postgres",
    password="1234"
)
THRESHOLD = 0.60      # 0.58-0.62 suele ir bien con face_recognition
# ===================

def get_pg_conn():
    return psycopg2.connect(**PG_CFG)

def cargar_base():
    """
    Lee (id, nombre, apellido, rostro BYTEA) desde PostgreSQL y
    calcula el embedding 128D de cada recorte guardado.
    Retorna: (embeddings Nx128 en np.array), (labels lista de str), (ids lista de int)
    """
    embs = []
    labels = []
    ids = []
    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, nombre, apellido, rostro FROM usuarios")
        rows = cur.fetchall()

    for uid, nombre, apellido, rostro_bytes in rows:
        npbuf = np.frombuffer(rostro_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)

        # El recorte es solo el rostro -> usar caja completa
        h, w = img_rgb.shape[:2]
        full_box = [(0, w, h, 0)]
        encs = face_recognition.face_encodings(img_rgb, known_face_locations=full_box, num_jitters=1)
        if not encs:
            # fallback por si el recorte no es perfecto
            locs = face_recognition.face_locations(img_rgb, model=("cnn" if USE_CNN else "hog"))
            if locs:
                encs = face_recognition.face_encodings(img_rgb, known_face_locations=locs, num_jitters=1)
        if encs:
            embs.append(np.array(encs[0], dtype=np.float32))
            labels.append(f"{nombre} {apellido}")
            ids.append(uid)

    if embs:
        embs = np.vstack(embs)  # (N,128)
    else:
        embs = np.empty((0,128), dtype=np.float32)
    return embs, labels, ids

def etiquetar_caras(frame_bgr, base_embs, base_labels):
    """
    Detecta caras en el frame y devuelve boxes (en coords del frame original) y labels.
    Hace la deteccion/encoding en una imagen reducida para ganar FPS y luego reescala.
    """
    # reducir y convertir a RGB
    small = cv2.resize(frame_bgr, None, fx=DOWNSCALE, fy=DOWNSCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # detectar caras en imagen reducida
    model = "cnn" if USE_CNN else "hog"
    boxes_small = face_recognition.face_locations(rgb_small, model=model)

    labels = []
    if len(boxes_small) > 0:
        # encodings sobre la misma imagen (rgb_small) y esas boxes
        encs = face_recognition.face_encodings(rgb_small, known_face_locations=boxes_small, num_jitters=1)

        for enc in encs:
            label = "Desconocido"
            if base_embs.shape[0] > 0:
                # distancias vectorizadas
                dists = face_recognition.face_distance(base_embs, enc)
                idx = int(np.argmin(dists))
                if dists[idx] <= THRESHOLD:
                    label = base_labels[idx]
            labels.append(label)
    else:
        encs = []

    # reescalar boxes al tamaño original
    scale = 1.0 / DOWNSCALE
    boxes = [(int(t*scale), int(r*scale), int(b*scale), int(l*scale)) for (t, r, b, l) in boxes_small]

    return boxes, labels

def main():
    print("Cargando base desde PostgreSQL...")
    base_embs, base_labels, base_ids = cargar_base()
    print(f"Embeddings cargados: {len(base_labels)}")
    if len(base_labels) == 0:
        print("ATENCION: no hay usuarios cargados. Registra alguno primero.")

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara. Cambia CAM_INDEX si es necesario.")

    print("Controles: 'r' recargar BD  |  'q' salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes, labels = etiquetar_caras(frame, base_embs, base_labels)

        # dibujar en la imagen original
        for (top, right, bottom, left), label in zip(boxes, labels):
            # cuadro
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
            # fondo del texto
            text = label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (left, top - th - 8), (left + tw + 6, top), (0, 200, 0), -1)
            # texto
            cv2.putText(frame, text, (left + 3, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Reconocimiento en tiempo real", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            print("Recargando base desde PostgreSQL...")
            base_embs, base_labels, base_ids = cargar_base()
            print(f"Embeddings cargados: {len(base_labels)}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
