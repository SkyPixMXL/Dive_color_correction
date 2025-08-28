import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os, math, threading, shutil, tempfile, subprocess
import numpy as np
import cv2

# ================== PARAMÈTRES GLOBAUX ==================
BLEND_ALPHA = 0.7
THRESHOLD_RATIO =5000
MIN_AVG_RED = 60
MAX_HUE_SHIFT = 120
BLUE_MAGIC_VALUE = 1.2
SAMPLE_SECONDS = 2  # calcule un filtre toutes les N secondes

IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".tiff",".tif"}
VIDEO_EXTS = {".mp4",".mov",".mkv",".avi",".m4v"}

# ================== OUTILS ==================
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def safe_float(x, default=1.0):
    try:
        x = float(x)
        if np.isfinite(x) and x != 0:
            return x
    except Exception:
        pass
    return float(default)

def blend_with_original(original, corrected, alpha):
    """Mélange l'image corrigée avec l'originale pour garder un rendu naturel."""
    return cv2.addWeighted(corrected, alpha, original, 1-alpha, 0)

def choose_fourcc(test_path, fps, frame_size):
    """
    Essaie plusieurs codecs H.264, sinon fallback mp4v.
    Retourne (fourcc_str, ok)
    """
    # ordre de préférence
    candidates = ["avc1", "H264", "h264", "mp4v"]
    for cand in candidates:
        fourcc = cv2.VideoWriter_fourcc(*cand)
        vw = cv2.VideoWriter(test_path, fourcc, max(1.0, fps), frame_size)
        ok = vw.isOpened()
        vw.release()
        if ok:
            try:
                if os.path.exists(test_path):
                    os.remove(test_path)
            except Exception:
                pass
            return cand, True
    return "mp4v", False

# ================== COULEUR / CORRECTION ==================
def hue_shift_red(mat, h):
    mat = mat.astype(np.float32, copy=False)
    U = math.cos(h * math.pi / 180.0)
    W = math.sin(h * math.pi / 180.0)
    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]
    return np.dstack([r, g, b]).astype(np.float32)

def normalizing_interval(array):
    # array peut contenir des zéros -> on récupère indices non nuls
    arr = np.asarray(array, dtype=np.float32)
    nz = arr[arr > 0]
    if nz.size < 2:
        return 0.0, 255.0
    nz = np.unique(np.clip(nz, 0, 255))
    diffs = nz[1:] - nz[:-1]
    if diffs.size == 0:
        return float(nz[0]), float(nz[-1])
    i = int(np.argmax(diffs))
    low = float(nz[i])
    high = float(nz[i+1])
    if high <= low:
        return 0.0, 255.0
    return low, high

def apply_filter(mat, filt):
    mat = mat.astype(np.float32, copy=False)
    r, g, b = mat[..., 0], mat[..., 1], mat[..., 2]
    # filt est de taille 20
    r2 = r * filt[0] + g * filt[1] + b * filt[2] + filt[4] * 255.0
    g2 = g * filt[6] + filt[9] * 255.0
    b2 = b * filt[12] + filt[14] * 255.0
    out = np.dstack([r2, g2, b2])
    return np.clip(out, 0, 255).astype(np.uint8)

def get_filter_matrix(mat_rgb):
    # Travail sur copie réduite pour la robustesse/performances
    small = cv2.resize(mat_rgb, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    avg = np.array(cv2.mean(small)[:3], dtype=np.float32)
    new_avg_r = float(avg[0])
    hue_shift = 0
    # Monte le rouge moyen au minimum requis
    while new_avg_r < MIN_AVG_RED and hue_shift <= MAX_HUE_SHIFT:
        shifted = hue_shift_red(avg[np.newaxis, np.newaxis, :], hue_shift)
        new_avg_r = float(np.sum(shifted))
        hue_shift += 1

    # Applique le décalage de teinte au canal R
    shifted_mat = hue_shift_red(small, hue_shift)
    new_r = np.sum(shifted_mat, axis=2)
    new_r = np.clip(new_r, 0, 255)
    small[..., 0] = new_r

    # Histogrammes
    hist_r = cv2.calcHist([small.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([small.astype(np.uint8)], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([small.astype(np.uint8)], [2], None, [256], [0, 256])

    normalize_mat = np.zeros((256, 3), dtype=np.float32)
    threshold_level = (small.shape[0] * small.shape[1]) / float(THRESHOLD_RATIO)

    for x in range(256):
        if hist_r[x] < threshold_level:
            normalize_mat[x, 0] = x
        if hist_g[x] < threshold_level:
            normalize_mat[x, 1] = x
        if hist_b[x] < threshold_level:
            normalize_mat[x, 2] = x
    normalize_mat[255] = [255, 255, 255]

    rl, rh = normalizing_interval(normalize_mat[:, 0])
    gl, gh = normalizing_interval(normalize_mat[:, 1])
    bl, bh = normalizing_interval(normalize_mat[:, 2])

    shifted_unit = hue_shift_red(np.array([[[1, 1, 1]]], dtype=np.float32), hue_shift)[0, 0]
    sr, sg, sb = [float(x) for x in shifted_unit]

    # Gains (protégés)
    rg = safe_float(256.0 / max(1e-6, (rh - rl)), 1.0)
    gg = safe_float(256.0 / max(1e-6, (gh - gl)), 1.0)
    bg = safe_float(256.0 / max(1e-6, (bh - bl)), 1.0)

    rOff = (-rl / 256.0) * rg
    gOff = (-gl / 256.0) * gg
    bOff = (-bl / 256.0) * bg

    adjust_red       = sr * rg
    adjust_red_green = sg * rg
    adjust_red_blue  = sb * rg * BLUE_MAGIC_VALUE

    return np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, rOff,
        0, gg, 0, 0, gOff,
        0, 0, bg, 0, bOff,
        0, 0, 0, 1, 0
    ], dtype=np.float32)

def correct_rgb_to_bgr(mat_bgr):
    rgb = cv2.cvtColor(mat_bgr, cv2.COLOR_BGR2RGB)
    filt = get_filter_matrix(rgb)
    corrected_rgb = apply_filter(rgb, filt)
    bgr=cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
    bgr = blend_with_original(mat_bgr, bgr, BLEND_ALPHA)
    return bgr

# ================== TRAITEMENT IMAGES ==================
def process_image(path, preview_cb=None):
    base, ext = os.path.splitext(path)
    out_path = f"{base}_T{ext}"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Impossible de lire l'image : {path}")
    corrected = correct_rgb_to_bgr(img)
    cv2.imwrite(out_path, corrected)
    if preview_cb:
        preview_cb(corrected)
    return out_path

# ================== TRAITEMENT VIDÉO ==================
def analyze_video(input_path, fps, frame_count):
    """Échantillonne des matrices de filtres à intervalle régulier."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {input_path}")

    step = max(1, int(round(fps * SAMPLE_SECONDS)))
    filters = []
    indices = []

    count = 0
    ok, frame = cap.read()
    while ok:
        count += 1
        if count % step == 0 or count == 1:
            filt = get_filter_matrix(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            filters.append(filt)
            indices.append(count)
        ok, frame = cap.read()

    cap.release()

    # Toujours au moins un filtre (au pire, milieu de la vidéo)
    if not filters:
        mid = max(1, frame_count // 2)
        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid-1)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Impossible d'échantillonner la vidéo.")
        filt = get_filter_matrix(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        filters = [filt]
        indices = [mid]

    filters = np.asarray(filters, dtype=np.float32)  # shape: (N, 20)
    indices = np.asarray(indices, dtype=np.float32)
    return filters, indices

def process_video_with_audio(input_path, progress_cb=None):
    """
    1) Corrige la vidéo frame par frame → temp mp4 (sans audio)
    2) Fusionne l'audio de la vidéo originale via FFmpeg → *_T.mp4
    3) Nettoie les fichiers temporaires
    """
    if not has_ffmpeg():
        raise RuntimeError("FFmpeg est requis pour l'audio (installe-le : brew install ffmpeg)")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0  # fallback raisonnable

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base, _ = os.path.splitext(input_path)
    final_out = f"{base}_T.mp4"

    # Fichier temporaire vidéo (corrigée, sans audio)
    tmp_dir = tempfile.mkdtemp(prefix="colfix_")
    temp_video = os.path.join(tmp_dir, "video_no_audio.mp4")

    # Choix du codec compatible
    test_path = os.path.join(tmp_dir, "probe.mp4")
    codec, _ = choose_fourcc(test_path, fps, (width, height))
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(temp_video, fourcc, float(fps), (width, height))
    if not out.isOpened():
        raise RuntimeError("Échec de l'ouverture du VideoWriter (codec non supporté).")

    # Préparer filtres (interpolation)
    filters, indices = analyze_video(input_path, fps, total)
    F = filters.shape[0]
    L = filters.shape[1]  # 20

    def interp_filter(n):
        # Interpolation 1D sur chaque coefficient
        vals = [np.interp(n, indices, filters[:, k]) for k in range(L)]
        return np.asarray(vals, dtype=np.float32)

    # Parcours des frames
    idx = 0
    ok, frame = cap.read()
    while ok:
        idx += 1
        fmat = interp_filter(idx) if F >= 2 else filters[0]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        corrected = apply_filter(rgb, fmat)
        corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
        out.write(corrected_bgr)

        if progress_cb and total > 0:
            progress_cb(idx / total)

        ok, frame = cap.read()

    cap.release()
    out.release()

    # Fusion vidéo corrigée + audio original
    # -c:v copy : copie la vidéo corrigée telle quelle (pas de ré-encodage)
    # -c:a aac  : audio ré-encodé proprement si nécessaire
    # -shortest : s'aligne sur la plus courte piste
    # -movflags +faststart : meilleur démarrage en streaming/QuickLook
    cmd = [
        "ffmpeg","-y",
        "-i", temp_video,
        "-i", input_path,
        "-map","0:v:0",
        "-map","1:a:0",
        "-c:v","copy",
        "-c:a","aac",
        "-movflags","+faststart",
        "-shortest",
        final_out
    ]

    # Si la vidéo source n'a pas d'audio, on ne mappe pas l'audio
    probe = subprocess.run(
        ["ffmpeg","-v","error","-i", input_path, "-map","0:a:0","-f","null","-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if probe.returncode != 0:
        # Pas d'audio détecté -> on sort juste la vidéo corrigée
        cmd = [
            "ffmpeg","-y",
            "-i", temp_video,
            "-c:v","copy",
            "-movflags","+faststart",
            final_out
        ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    # Nettoyage
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    return final_out

# ================== GUI ==================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Correction couleur (avec audio)")
        self.geometry("720x560")

        frm = tk.Frame(self)
        frm.pack(padx=10, pady=10, fill="both", expand=True)

        self.btn_browse = tk.Button(frm, text="Sélectionner fichiers", command=self.browse_files)
        self.btn_browse.pack(fill=tk.X)

        self.listbox = tk.Listbox(frm, width=100, height=12, selectmode=tk.MULTIPLE)
        self.listbox.pack(pady=8, fill="both", expand=True)

        self.preview_label = tk.Label(frm, relief="groove", height=12)
        self.preview_label.pack(fill=tk.BOTH, pady=6)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(frm, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=6)

        self.status = tk.StringVar(value="Prêt.")
        tk.Label(frm, textvariable=self.status, anchor="w").pack(fill=tk.X)

        self.btn_start = tk.Button(frm, text="Lancer correction", command=self.start_thread)
        self.btn_start.pack(fill=tk.X, pady=4)

    def browse_files(self):
        files = filedialog.askopenfilenames(
            title="Sélectionner des images/vidéos",
            filetypes=[("Images et vidéos","*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.mp4 *.mov *.mkv *.avi *.m4v")]
        )
        self.listbox.delete(0, tk.END)
        for p in files:
            self.listbox.insert(tk.END, p)

    # Mises à jour thread-safe
    def set_status(self, text):
        self.status.set(text)
        self.update_idletasks()

    def set_progress(self, frac):
        self.progress_var.set(max(0, min(100, float(frac)*100.0)))
        self.update_idletasks()

    def show_preview(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail((480, 320))
        imgtk = ImageTk.PhotoImage(im)
        self.preview_label.configure(image=imgtk)
        self.preview_label.image = imgtk

    def start_thread(self):
        if self.listbox.size() == 0:
            messagebox.showwarning("Aucune sélection", "Sélectionne au moins un fichier.")
            return
        t = threading.Thread(target=self.run_all, daemon=True)
        self.btn_start.config(state="disabled")
        self.set_progress(0)
        t.start()

    def run_all(self):
        # Filtrage et traitement
        if not has_ffmpeg():
            self.after(0, lambda: messagebox.showwarning(
                "FFmpeg manquant",
                "FFmpeg est requis pour l'audio.\nInstalle-le avec : brew install ffmpeg"
            ))

        total_files = self.listbox.size()
        for i in range(total_files):
            path = self.listbox.get(i)
            ext = os.path.splitext(path)[1].lower()

            def ui_progress(frac):
                self.after(0, lambda f=frac: self.set_progress( (i + f) / total_files ))

            self.after(0, lambda p=path: self.set_status(f"Traitement : {p}"))

            try:
                if ext in IMAGE_EXTS:
                    out = process_image(path, preview_cb=lambda img: self.after(0, lambda: self.show_preview(img)))
                    self.after(0, lambda: self.set_status(f"Image corrigée : {out}"))
                    self.after(0, lambda: self.set_progress((i+1)/total_files))

                elif ext in VIDEO_EXTS:
                    # l’aperçu live vidéo coûteux : on met à jour juste la barre
                    out = process_video_with_audio(path, progress_cb=ui_progress)
                    self.after(0, lambda: self.set_status(f"Vidéo finale (audio conservé) : {out}"))
                else:
                    self.after(0, lambda: self.set_status(f"Format non supporté : {path}"))

            except Exception as e:
                self.after(0, lambda: self.set_status(f"Erreur: {e}"))

        self.after(0, lambda: self.set_progress(1.0))
        self.after(0, lambda: messagebox.showinfo("Terminé", "Correction terminée pour tous les fichiers."))
        self.after(0, lambda: self.btn_start.config(state="normal"))

if __name__ == "__main__":
    App().mainloop()
