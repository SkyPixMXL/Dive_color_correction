import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import math
import threading
import queue
import os

# MoviePy est nécessaire pour le traitement vidéo avec audio
try:
    import moviepy.editor as mp
except ImportError:
    messagebox.showerror("Dépendance Manquante", "La bibliothèque 'moviepy' est requise pour le traitement vidéo.\n\nVeuillez l'installer en utilisant la commande :\npip install moviepy")
    exit()

# -----------------------------------------------------------------------------
# SECTION 1: LOGIQUE DE TRAITEMENT D'IMAGE
# -----------------------------------------------------------------------------
# Ces fonctions sont votre algorithme de base pour la correction des couleurs.
# Elles sont appelées par l'interface graphique pour traiter les images.

def hue_shift_red(mat, h):
    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)
    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]
    return np.dstack([r, g, b])

def normalizing_interval(array):
    high, low, max_dist = 255, 0, 0
    for i in range(1, len(array)):
        dist = array[i] - array[i-1]
        if dist > max_dist:
            max_dist, high, low = dist, array[i], array[i-1]
    return (low, high)

def apply_filter(mat, filt):
    r = mat[..., 0] * filt[0] + mat[..., 1] * filt[1] + mat[..., 2] * filt[2] + filt[4] * 255
    g = mat[..., 1] * filt[6] + filt[9] * 255
    b = mat[..., 2] * filt[12] + filt[14] * 255
    filtered_mat = np.dstack([r, g, b])
    return np.clip(filtered_mat, 0, 255).astype(np.uint8)

def get_filter_matrix(mat, params):
    mat = cv2.resize(mat, (256, 256))
    avg_mat = np.array(cv2.mean(mat)[:3], dtype=np.uint8)
    new_avg_r, hue_shift = avg_mat[0], 0
    while new_avg_r < params['MIN_AVG_RED']:
        shifted = hue_shift_red(avg_mat, hue_shift)
        new_avg_r = np.sum(shifted)
        hue_shift += 1
        if hue_shift > params['MAX_HUE_SHIFT']:
            new_avg_r = params['MIN_AVG_RED']
    
    shifted_mat = hue_shift_red(mat, hue_shift)
    new_r_channel = np.clip(np.sum(shifted_mat, axis=2), 0, 255)
    mat[..., 0] = new_r_channel

    hist_r = cv2.calcHist([mat], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([mat], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([mat], [2], None, [256], [0, 256])

    normalize_mat = np.zeros((256, 3))
    threshold_level = (mat.shape[0] * mat.shape[1]) / params['THRESHOLD_RATIO']
    for x in range(256):
        if hist_r[x] < threshold_level: normalize_mat[x][0] = x
        if hist_g[x] < threshold_level: normalize_mat[x][1] = x
        if hist_b[x] < threshold_level: normalize_mat[x][2] = x
    normalize_mat[255] = [255, 255, 255]

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])

    shifted_r, shifted_g, shifted_b = hue_shift_red(np.array([1, 1, 1]), hue_shift)[0][0]

    red_gain = 256 / (adjust_r_high - adjust_r_low) if (adjust_r_high - adjust_r_low) != 0 else 0
    green_gain = 256 / (adjust_g_high - adjust_g_low) if (adjust_g_high - adjust_g_low) != 0 else 0
    blue_gain = 256 / (adjust_b_high - adjust_b_low) if (adjust_b_high - adjust_b_low) != 0 else 0

    redOffset = (-adjust_r_low / 256) * red_gain
    greenOffset = (-adjust_g_low / 256) * green_gain
    blueOffset = (-adjust_b_low / 256) * blue_gain

    adjust_red = shifted_r * red_gain
    adjust_red_green = shifted_g * red_gain
    adjust_red_blue = shifted_b * red_gain * params['BLUE_MAGIC_VALUE']

    return np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        0, green_gain, 0, 0, greenOffset,
        0, 0, blue_gain, 0, blueOffset,
        0, 0, 0, 1, 0,
    ])

# -----------------------------------------------------------------------------
# SECTION 2: CLASSE PRINCIPALE DE L'APPLICATION TKINTER
# -----------------------------------------------------------------------------

class DivingMediaCorrector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Correcteur Média Sous-Marin (Tkinter)")
        self.geometry("1200x800")

        # Variables d'état
        self.original_mat_bgr = None
        self.corrected_mat_bgr = None
        self.is_video = False
        self.current_filepath = None
        self.loaded_files = [] 
        self.preview_photo = None

        self.default_params = {
            'THRESHOLD_RATIO': 2000,
            'MIN_AVG_RED': 60,
            'MAX_HUE_SHIFT': 120,
            'BLUE_MAGIC_VALUE': 1.2
        }

        self.thread_queue = queue.Queue()
        self.create_widgets()
        self.after(100, self.check_thread_queue)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        files_frame = ttk.LabelFrame(main_frame, text="Galerie de Fichiers", padding="5")
        files_frame.grid(row=0, column=0, sticky="ew", pady=5)
        files_frame.grid_columnconfigure(0, weight=1)

        list_frame = ttk.Frame(files_frame)
        list_frame.grid(row=0, column=0, sticky="ew")
        list_frame.grid_columnconfigure(0, weight=1)
        
        self.file_listbox = tk.Listbox(list_frame, height=5)
        self.file_listbox.grid(row=0, column=0, sticky="ew")
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.file_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_listbox['yscrollcommand'] = scrollbar.set

        buttons_frame = ttk.Frame(files_frame)
        buttons_frame.grid(row=1, column=0, sticky="ew")
        
        add_button = ttk.Button(buttons_frame, text="Ajouter des Fichiers...", command=self.add_files)
        add_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        clear_button = ttk.Button(buttons_frame, text="Vider la Liste", command=self.clear_file_list)
        clear_button.pack(side=tk.LEFT, padx=5, pady=5)

        center_frame = ttk.Frame(main_frame)
        center_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        center_frame.grid_columnconfigure(0, weight=3)
        center_frame.grid_columnconfigure(1, weight=1)
        center_frame.grid_rowconfigure(0, weight=1)

        self.image_label = ttk.Label(center_frame, text="L'aperçu apparaîtra ici", anchor=tk.CENTER, background="gray")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=5)

        params_frame = ttk.LabelFrame(center_frame, text="Paramètres du Filtre", padding="10")
        params_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        self.create_param_sliders(params_frame)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, sticky="ew", pady=5)

        self.status_label = ttk.Label(bottom_frame, text="Prêt")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(bottom_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, expand=True, padx=5)

    def create_param_sliders(self, parent_frame):
        self.params = {key: tk.DoubleVar(value=val) for key, val in self.default_params.items()}
        
        def create_slider(label, key, from_, to):
            ttk.Label(parent_frame, text=label).pack(fill=tk.X, pady=(10, 0))
            slider = ttk.Scale(parent_frame, from_=from_, to=to, variable=self.params[key], orient='horizontal', command=lambda e: self.update_preview())
            slider.pack(fill=tk.X)
        
        create_slider("Ratio Seuil", 'THRESHOLD_RATIO', 500, 5000)
        create_slider("Rouge Moyen Min", 'MIN_AVG_RED', 30, 100)
        create_slider("Décalage Teinte Max", 'MAX_HUE_SHIFT', 50, 200)
        create_slider("Magie Bleue", 'BLUE_MAGIC_VALUE', 0.5, 2.0)

        self.save_button = ttk.Button(parent_frame, text="Enregistrer l'Image Corrigée", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, pady=20, expand=True)

        reset_button = ttk.Button(parent_frame, text="Réinitialiser", command=self.reset_parameters)
        reset_button.pack(side=tk.RIGHT, pady=20, expand=True)

    def add_files(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("Fichiers Média", "*.jpg *.jpeg *.png *.mp4 *.mov *.avi")])
        if not filepaths: return
        
        for fp in filepaths:
            if fp not in self.loaded_files:
                self.loaded_files.append(fp)
                self.file_listbox.insert(tk.END, os.path.basename(fp))
        
        if self.file_listbox.curselection() == ():
            self.file_listbox.selection_set(0)
            self.on_file_select(None)

    def clear_file_list(self):
        self.file_listbox.delete(0, tk.END)
        self.loaded_files.clear()
        self.original_mat_bgr = None
        self.corrected_mat_bgr = None
        self.current_filepath = None
        self.image_label.config(image='', text="L'aperçu apparaîtra ici")
        self.save_button.config(state=tk.DISABLED)
        self.status_label.config(text="Prêt")
        self.progress_bar['value'] = 0

    def on_file_select(self, event):
        selection_indices = self.file_listbox.curselection()
        if not selection_indices: return
        
        selected_index = selection_indices[0]
        self.current_filepath = self.loaded_files[selected_index]
        self.is_video = any(self.current_filepath.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi'])

        if self.is_video:
            self.save_button.config(state=tk.DISABLED)
            cap = cv2.VideoCapture(self.current_filepath)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.original_mat_bgr = frame
                self.update_preview()
                self.process_video()
            else:
                messagebox.showerror("Erreur", "Impossible de lire la première image de la vidéo.")
        else:
            self.original_mat_bgr = cv2.imread(self.current_filepath)
            if self.original_mat_bgr is not None:
                self.save_button.config(state=tk.NORMAL)
                self.update_preview()
            else:
                messagebox.showerror("Erreur", "Impossible de charger le fichier image.")

    def update_preview(self):
        if self.original_mat_bgr is None: return

        current_params = {key: var.get() for key, var in self.params.items()}
        rgb_mat = cv2.cvtColor(self.original_mat_bgr, cv2.COLOR_BGR2RGB)
        filter_matrix = get_filter_matrix(rgb_mat, current_params)
        corrected_mat_rgb = apply_filter(rgb_mat, filter_matrix)
        self.corrected_mat_bgr = cv2.cvtColor(corrected_mat_rgb, cv2.COLOR_RGB2BGR)

        preview_bgr = self.original_mat_bgr.copy()
        width = preview_bgr.shape[1] // 2
        preview_bgr[:, width:] = self.corrected_mat_bgr[:, width:]

        h, w = preview_bgr.shape[:2]
        max_h, max_w = self.image_label.winfo_height(), self.image_label.winfo_width()
        if max_h < 2 or max_w < 2: max_h, max_w = 540, 960
        scale = min(max_w / w, max_h / h) if w > 0 and h > 0 else 1
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_preview = cv2.resize(preview_bgr, (new_w, new_h))
        img = cv2.cvtColor(resized_preview, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        self.preview_photo = ImageTk.PhotoImage(image=img_pil)
        self.image_label.config(image=self.preview_photo)

    def reset_parameters(self):
        for key, value in self.default_params.items():
            self.params[key].set(value)
        self.update_preview()

    def save_image(self):
        if self.corrected_mat_bgr is None or self.is_video: return
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Image PNG", "*.png"), ("Image JPEG", "*.jpg")])
        if save_path:
            cv2.imwrite(save_path, self.corrected_mat_bgr)
            messagebox.showinfo("Succès", f"Image enregistrée sous {os.path.basename(save_path)}")

    def process_video(self):
        output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("Vidéo MP4", "*.mp4")])
        if not output_path: return
        
        current_params = {key: var.get() for key, var in self.params.items()}
        current_params['SAMPLE_SECONDS'] = 2
        
        thread = threading.Thread(target=self.video_processing_thread, args=(self.current_filepath, output_path, current_params), daemon=True)
        thread.start()

    def video_processing_thread(self, input_path, output_path, params):
        try:
            # --- Phase 1: Analyse avec OpenCV (rapide) ---
            self.thread_queue.put(('status', f'Analyse de {os.path.basename(input_path)}...'))
            cap = cv2.VideoCapture(input_path)
            fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            filter_matrices, filter_indices, count = [], [], 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                count += 1
                self.thread_queue.put(('progress', (count, frame_count)))
                if count % (fps * params['SAMPLE_SECONDS']) == 0:
                    mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    filter_indices.append(count)
                    filter_matrices.append(get_filter_matrix(mat, params))
            cap.release()
            if not filter_matrices:
                 self.thread_queue.put(('error', 'Vidéo trop courte ou erreur de lecture.'))
                 return

            # --- Phase 2: Traitement avec MoviePy (conserve l'audio) ---
            self.thread_queue.put(('status', f'Correction et encodage de {os.path.basename(input_path)}... Patientez.'))
            self.thread_queue.put(('indeterminate_start', None))

            clip = mp.VideoFileClip(input_path)
            
            filter_matrices_np = np.array(filter_matrices)
            filter_indices_np = np.array(filter_indices)
            
            def get_interpolated_filter_matrix(frame_num):
                if frame_num <= filter_indices_np[0]: return filter_matrices_np[0]
                if frame_num >= filter_indices_np[-1]: return filter_matrices_np[-1]
                return [np.interp(frame_num, filter_indices_np, filter_matrices_np[..., x]) for x in range(len(filter_matrices_np[0]))]

            def transform_frame(frame_at_t, t):
                frame_num = int(t * clip.fps)
                interpolated_filter = get_interpolated_filter_matrix(frame_num)
                # MoviePy utilise des frames RGB, ce qui est attendu par apply_filter
                return apply_filter(frame_at_t, interpolated_filter)

            processed_clip = clip.fl(lambda gf, t: transform_frame(gf(t), t))
            
            # Réassigner l'audio original au clip traité
            if clip.audio:
                processed_clip.audio = clip.audio
            
            processed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)
            
            clip.close()
            processed_clip.close()
            
            self.thread_queue.put(('done', f'Terminé ! Vidéo enregistrée sous {os.path.basename(output_path)}'))
        except Exception as e:
            self.thread_queue.put(('error', f'Erreur lors du traitement : {e}'))

    def check_thread_queue(self):
        try:
            message = self.thread_queue.get(block=False)
            msg_type, data = message

            if msg_type == 'status':
                self.status_label.config(text=data)
            elif msg_type == 'progress':
                current, total = data
                self.progress_bar.config(mode='determinate')
                self.progress_bar['maximum'] = total
                self.progress_bar['value'] = current
            elif msg_type == 'indeterminate_start':
                self.progress_bar.config(mode='indeterminate')
                self.progress_bar.start(10)
            elif msg_type == 'done':
                self.progress_bar.stop()
                self.progress_bar.config(mode='determinate')
                self.status_label.config(text="Prêt")
                self.progress_bar['value'] = 0
                messagebox.showinfo("Succès", data)
            elif msg_type == 'error':
                self.progress_bar.stop()
                self.progress_bar.config(mode='determinate')
                self.status_label.config(text="Erreur")
                self.progress_bar['value'] = 0
                messagebox.showerror("Erreur", data)

        except queue.Empty:
            pass
        finally:
            self.after(100, self.check_thread_queue)

if __name__ == "__main__":
    app = DivingMediaCorrector()
    app.mainloop()
