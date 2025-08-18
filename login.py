import cv2
import face_recognition
import numpy as np
import psycopg2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog  # <-- NEW: simpledialog
from PIL import Image, ImageTk
import threading
import time

# ====== IMPORTA LA FUNCIÓN DEL PRIMER ARCHIVO ======
# Cambia 'registrar' por el nombre real de tu primer archivo si es distinto.
from detectar_rostros import insertar_usuario  # <-- NEW

# ========= Config =========
USE_CNN         = False     # HOG (False) es mas rapido en CPU
CAM_INDEX       = 0
CAP_WIDTH       = 640
CAP_HEIGHT      = 480
DOWNSCALE       = 0.5
UPSAMPLE        = 0
JITTERS         = 0
THRESHOLD       = 0.60
PROCESS_EVERY_N = 2         # procesa 1 de cada N frames
REQUIRED_HITS   = 5         # aciertos seguidos para “loguear”

PG_CFG = dict(
    host="dpg-d2gdo20dl3ps73f7ic5g-a.oregon-postgres.render.com",
    port=5432,
    dbname="reconocimiento_6ezo",
    user="usr_recon",
    password="hXcXXHnG17XXr4RDRs3MLmHtFQnM7BA7",
    sslmode="require",   # Render exige SSL
)
# ==========================

def get_pg_conn():
    return psycopg2.connect(
        host="dpg-d2gdo20dl3ps73f7ic5g-a.oregon-postgres.render.com",
        port=5432,
        dbname="reconocimiento_6ezo",
        user="usr_recon",
        password="hXcXXHnG17XXr4RDRs3MLmHtFQnM7BA7",
        sslmode="require"
    )

def cargar_base_desde_bd():
    """Lee usuarios (id, nombre, apellido, rostro BYTEA) -> (embs Nx128), (labels), (ids)."""
    embs, labels, ids = [], [], []
    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, nombre, apellido, rostro FROM usuarios")
        rows = cur.fetchall()

    for uid, nombre, apellido, rostro_bytes in rows:
        if not rostro_bytes:
            continue
        npbuf = np.frombuffer(rostro_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        h, w = img_rgb.shape[:2]
        full_box = [(0, w, h, 0)]
        encs = face_recognition.face_encodings(img_rgb, known_face_locations=full_box, num_jitters=0)
        if not encs:
            locs = face_recognition.face_locations(img_rgb, model=("cnn" if USE_CNN else "hog"),
                                                   number_of_times_to_upsample=0)
            if locs:
                encs = face_recognition.face_encodings(img_rgb, known_face_locations=locs, num_jitters=0)
        if encs:
            embs.append(np.array(encs[0], dtype=np.float32))
            labels.append(f"{nombre} {apellido}")
            ids.append(uid)

    if embs:
        embs = np.vstack(embs).astype(np.float32)
    else:
        embs = np.empty((0, 128), dtype=np.float32)
    return embs, labels, ids

class FaceLoginApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Login por Reconocimiento Facial")
        self.geometry("800x560")
        self.resizable(False, False)

        # Estado
        self.base_embs, self.base_labels, self.base_ids = cargar_base_desde_bd()
        self.cap = None
        self.stop_flag = False
        self.frame_counter = 0
        self.last_boxes = []
        self.last_labels = []
        self.hit_label = None
        self.hit_count = 0
        self.mode = "cam"  # "cam" o "welcome"

        # Contenedor/pantallas
        self.container = ttk.Frame(self, padding=8)
        self.container.pack(fill="both", expand=True)

        self.screen_cam = ttk.Frame(self.container)
        self.screen_welcome = ttk.Frame(self.container)

        # ---- Pantalla Cam ----
        self.video_label = ttk.Label(self.screen_cam)
        self.video_label.pack()

        controls = ttk.Frame(self.screen_cam)
        controls.pack(fill="x", pady=6)

        # Botón NUEVO: Registrar usuario
        ttk.Button(controls, text="Registrar usuario", command=self.on_register).pack(side="left", padx=6)  # <-- NEW

        ttk.Button(controls, text="Recargar BD", command=self.reload_db).pack(side="left", padx=6)
        ttk.Button(controls, text="Salir", command=self.on_close).pack(side="right")

        hint = ttk.Label(self.screen_cam, text="Coloca tu rostro frente a la camara para iniciar sesion...")
        hint.pack(pady=4)

        # ---- Pantalla Bienvenida ----
        self.welcome_label = ttk.Label(self.screen_welcome, font=("Segoe UI", 20, "bold"))
        self.welcome_label.pack(pady=40)

        btns = ttk.Frame(self.screen_welcome)
        btns.pack(pady=10)
        ttk.Button(btns, text="Cerrar sesion", command=self.on_logout).pack(side="left", padx=6)
        ttk.Button(btns, text="Cerrar", command=self.on_close).pack(side="left", padx=6)

        # Mostrar pantalla cam
        self.show_screen(self.screen_cam)

        # Iniciar camara/hilo
        self.init_camera()
        self.video_thread = threading.Thread(target=self.loop_camera, daemon=True)
        self.video_thread.start()

        # Protocol cierre ventana
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_screen(self, frame):
        for child in self.container.winfo_children():
            child.pack_forget()
        frame.pack(fill="both", expand=True)

    def reload_db(self):
        self.base_embs, self.base_labels, self.base_ids = cargar_base_desde_bd()
        messagebox.showinfo("Base recargada", f"Registros cargados: {len(self.base_labels)}")

    def init_camera(self):
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la camara.")
            self.destroy()
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    def loop_camera(self):
        while not self.stop_flag:
            # Si estamos en welcome, no procesar (ahorra CPU)
            if self.mode == "welcome":
                time.sleep(0.05)
                continue

            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # Procesar 1 de cada N frames
            if self.frame_counter % PROCESS_EVERY_N == 0:
                boxes, labels = self.detect_and_label(frame)
                self.last_boxes, self.last_labels = boxes, labels
                # Conteo de aciertos seguidos
                candidate = None
                if labels:
                    candidate = labels[0] if labels[0] != "Desconocido" else None
                if candidate and candidate == self.hit_label:
                    self.hit_count += 1
                elif candidate:
                    self.hit_label = candidate
                    self.hit_count = 1
                else:
                    self.hit_label = None
                    self.hit_count = 0

                if self.hit_label and self.hit_count >= REQUIRED_HITS:
                    self.show_welcome(self.hit_label)
                    # Cambiamos modo, pero NO cerramos camara/hilo
                    self.mode = "welcome"
                    # reset de dibujo/estado para cuando volvamos
                    self.last_boxes, self.last_labels = [], []
                    self.hit_label = None
                    self.hit_count = 0

            self.frame_counter += 1

            # Dibujar
            vis = frame.copy()
            for (top, right, bottom, left), text in zip(self.last_boxes, self.last_labels):
                cv2.rectangle(vis, (left, top), (right, bottom), (0, 200, 0), 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y0 = max(top - th - 8, 0)
                cv2.rectangle(vis, (left, y0), (left + tw + 6, y0 + th + 8), (0, 200, 0), -1)
                cv2.putText(vis, text, (left + 3, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Mostrar en Tkinter (solo en modo cam)
            if self.mode == "cam":
                rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))

                # UI update seguro desde hilo principal
                def update_ui():
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
                self.after(0, update_ui)

        # liberar cam al salir
        if self.cap is not None:
            self.cap.release()

    def detect_and_label(self, frame_bgr):
        small = cv2.resize(frame_bgr, None, fx=DOWNSCALE, fy=DOWNSCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        model = "cnn" if USE_CNN else "hog"
        boxes_small = face_recognition.face_locations(
            rgb_small, model=model, number_of_times_to_upsample=UPSAMPLE
        )
        labels = []
        if boxes_small:
            encs = face_recognition.face_encodings(
                rgb_small, known_face_locations=boxes_small, num_jitters=JITTERS
            )
            for enc in encs:
                label = "Desconocido"
                if self.base_embs.shape[0] > 0:
                    dists = face_recognition.face_distance(self.base_embs, enc)
                    idx = int(np.argmin(dists))
                    if dists[idx] <= THRESHOLD:
                        label = self.base_labels[idx]
                labels.append(label)

        scale = 1.0 / DOWNSCALE
        boxes = [(int(t*scale), int(r*scale), int(b*scale), int(l*scale)) for (t, r, b, l) in boxes_small]
        return boxes, labels

    # ============ NUEVO: Registrar usuario ============
    def on_register(self):
        """
        Hace un registro estilo 'primer archivo' pero dentro de esta misma ventana:
         - toma el frame actual
         - detecta el primer rostro
         - pide datos (Nombre, Apellido, Usuario, Contrasena)
         - llama a insertar_usuario(...) del primer archivo
         - recarga la base en memoria
        """
        if self.cap is None:
            messagebox.showerror("Error", "Camara no inicializada.")
            return

        # Tomar un frame reciente y nítido
        ok, frame = self.cap.read()
        if not ok:
            messagebox.showerror("Error", "No se pudo leer frame de la camara.")
            return

        # Detectar rostros en frame completo (como en el primer archivo)
        small = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        model = "cnn" if USE_CNN else "hog"
        boxes_small = face_recognition.face_locations(rgb_small, model=model, number_of_times_to_upsample=UPSAMPLE)

        if not boxes_small:
            messagebox.showinfo("Registro", "No hay rostro claro para registrar. Acércate y mira a la cámara.")
            return

        # Tomar el primer rostro detectado y reescalar a tamaño original
        scale = 1.0 / DOWNSCALE
        t, r, b, l = boxes_small[0]
        top, right, bottom, left = int(t*scale), int(r*scale), int(b*scale), int(l*scale)

        # Validar recorte
        if bottom - top <= 0 or right - left <= 0:
            messagebox.showwarning("Registro", "Recorte vacío o incorrecto. Intenta de nuevo.")
            return

        rostro_bgr = frame[top:bottom, left:right]

        # Validación rápida: asegurar que haya encoding (igual que el primer archivo)
        rostro_rgb = cv2.cvtColor(rostro_bgr, cv2.COLOR_BGR2RGB)
        rostro_rgb = np.ascontiguousarray(rostro_rgb, dtype=np.uint8)
        h, w = rostro_rgb.shape[:2]
        full_box = [(0, w, h, 0)]
        encs = face_recognition.face_encodings(rostro_rgb, known_face_locations=full_box, num_jitters=1)
        if not encs:
            messagebox.showinfo("Registro", "No se pudo validar el rostro (sin encoding). Mejora la luz o reencuadra.")
            return

        # Pedir datos del usuario (como en el primer archivo, con defaults si se deja vacío)
        nombre = simpledialog.askstring("Registro", "Nombre:", parent=self) or "sin_nombre"
        apellido = simpledialog.askstring("Registro", "Apellido:", parent=self) or "sin_apellido"
        usuario = simpledialog.askstring("Registro", "Usuario:", parent=self) or "sin_usuario"
        contrasena = simpledialog.askstring("Registro", "Contrasena:", parent=self, show="*") or "sin_contrasena"

        # Guardar en BD usando la función del PRIMER ARCHIVO
        try:
            uid = insertar_usuario(nombre, apellido, usuario, contrasena, rostro_bgr)
            messagebox.showinfo("Registro", f"Registrado OK -> id={uid}\n{nombre} {apellido} (usuario: {usuario})")
            # Recargar base en memoria para que el nuevo aparezca ya disponible
            self.reload_db()
        except Exception as e:
            messagebox.showerror("Error al registrar", str(e))
    # ============ FIN NUEVO ============

    def show_welcome(self, label_text):
        self.welcome_label.config(text=f"Bienvenido: {label_text}")
        self.show_screen(self.screen_welcome)

    def on_logout(self):
        """Volver a la camara (cerrar sesion)."""
        # reset de contadores/estado
        self.hit_label = None
        self.hit_count = 0
        self.frame_counter = 0
        self.last_boxes, self.last_labels = [], []
        # modo cam
        self.mode = "cam"
        self.show_screen(self.screen_cam)

    def on_close(self):
        self.stop_flag = True
        self.after(100, self.destroy)

if __name__ == "__main__":
    app = FaceLoginApp()
    app.mainloop()
