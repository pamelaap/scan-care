import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pydicom
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import numpy as np
from skimage import measure, morphology, filters
from stl import mesh
import SimpleITK as sitk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from datetime import datetime
import matplotlib
from scipy.ndimage import gaussian_filter
matplotlib.use('TkAgg')

class AdvancedDICOMViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("DICOM Viewer 3D Avanzado con Segmentación Inteligente")
        self.root.geometry("1400x1000")
        self.root.option_add("*Font", "Montserrat 11")
        self.root.configure(bg="#e0f7fa")  # celeste claro
        ttk.Style().configure("TButton", background="#8ecae6")
        # Título principal
        title_label = tk.Label(self.root, text="Visor DICOM con Segmentación y Exportación 3D", font=("Montserrat", 16, "bold"), fg="#2c3e50")
        title_label.pack(pady=(15, 5))


        
        # Cargar logos y mostrar en la parte superior
        logo_frame = tk.Frame(self.root)
        logo_frame.pack(fill=tk.X)

        try:
            from PIL import Image, ImageTk
            self.logo_scan = Image.open("C:/Users/pamel/Downloads/lognopng (2).png").resize((150, 50))
            self.logo_tec = Image.open("C:/Users/pamel/Downloads/logotecnegro.png").resize((150, 50))
            self.tk_logo_scan = ImageTk.PhotoImage(self.logo_scan)
            self.tk_logo_tec = ImageTk.PhotoImage(self.logo_tec)
            tk.Label(logo_frame, image=self.tk_logo_scan).pack(side=tk.LEFT, padx=10, pady=5)
            tk.Label(logo_frame, image=self.tk_logo_tec).pack(side=tk.RIGHT, padx=10, pady=5)
        except Exception as e:
            print(f"No se pudieron cargar los logos: {e}")

        
        # Configuración inicial
        self.dicom_series = None
        self.pixel_array = None
        self.sitk_image = None
        self.current_slices = {'axial': 0, 'sagital': 0, 'coronal': 0}
        self.export_path = os.path.expanduser("~/Downloads/DICOM_Exports")
        self.orientation_corrected = False
        
        # Variables para segmentación
        self.threshold_min = tk.DoubleVar(value=300)
        self.threshold_max = tk.DoubleVar(value=4000)
        self.min_volume_size = tk.IntVar(value=1000)
        self.smoothing_value = tk.DoubleVar(value=1.0)
        self.invert_volume = tk.BooleanVar(value=False)
        self.segmentation_preset = tk.StringVar(value="Manual")
        self.presets = {
            "Manual": (None, None),
            "Hueso (300–3000 HU)": (300, 3000),
            "Tumor cerebral (40–80 HU)": (40, 80),
            "Tejido blando (-100–300 HU)": (-100, 300)
        }

        
        # Variables de visualización
        self.window_center = tk.IntVar(value=40)
        self.window_width = tk.IntVar(value=400)
        
        # Segmentación actual
        self.current_mask = None
        self.roi_selector = None
        
        self.create_ui()
        self.setup_shortcuts()
    
    def create_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel de visualización triple
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Vistas DICOM
        self.axial_frame = ttk.LabelFrame(display_frame, text="Vista Axial (Selección ROI)")
        self.axial_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.sagital_frame = ttk.LabelFrame(display_frame, text="Vista Sagital")
        self.sagital_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.coronal_frame = ttk.LabelFrame(display_frame, text="Vista Coronal")
        self.coronal_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        
        # Configurar grid para expansión uniforme
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_columnconfigure(1, weight=1)
        display_frame.grid_columnconfigure(2, weight=1)
        
        # Crear figuras y canvas para cada vista
        self.fig_axial, self.ax_axial = plt.subplots(figsize=(4, 4), dpi=100)
        self.canvas_axial = FigureCanvasTkAgg(self.fig_axial, master=self.axial_frame)
        self.canvas_axial_widget = self.canvas_axial.get_tk_widget()
        self.canvas_axial_widget.pack(fill=tk.BOTH, expand=True)
        
        self.fig_sagital, self.ax_sagital = plt.subplots(figsize=(4, 4), dpi=100)
        self.canvas_sagital = FigureCanvasTkAgg(self.fig_sagital, master=self.sagital_frame)
        self.canvas_sagital_widget = self.canvas_sagital.get_tk_widget()
        self.canvas_sagital_widget.pack(fill=tk.BOTH, expand=True)
        
        self.fig_coronal, self.ax_coronal = plt.subplots(figsize=(4, 4), dpi=100)
        self.canvas_coronal = FigureCanvasTkAgg(self.fig_coronal, master=self.coronal_frame)
        self.canvas_coronal_widget = self.canvas_coronal.get_tk_widget()
        self.canvas_coronal_widget.pack(fill=tk.BOTH, expand=True)
        
        # Panel de controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Menú de presets
        preset_frame = ttk.LabelFrame(control_frame, text="Presets de Umbral")
        preset_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(preset_frame, text="Estructura:").grid(row=0, column=0, padx=2)
        preset_menu = ttk.OptionMenu(preset_frame, self.segmentation_preset, "Manual", *self.presets.keys(), command=self.apply_preset)
        preset_menu.grid(row=0, column=1, padx=2)


        # Controles de navegación
        nav_frame = ttk.LabelFrame(control_frame, text="Navegación")
        nav_frame.pack(side=tk.LEFT, padx=5)
        
        # Sliders para cada vista
        ttk.Label(nav_frame, text="Axial:").grid(row=0, column=0, padx=2)
        self.axial_slider = ttk.Scale(nav_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                    command=lambda v: self.on_slice_change('axial', v))
        self.axial_slider.grid(row=0, column=1, padx=5, pady=2)
        self.axial_label = ttk.Label(nav_frame, text="Slice: 0/0")
        self.axial_label.grid(row=0, column=2, padx=2)
        
        ttk.Label(nav_frame, text="Sagital:").grid(row=1, column=0, padx=2)
        self.sagital_slider = ttk.Scale(nav_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                      command=lambda v: self.on_slice_change('sagital', v))
        self.sagital_slider.grid(row=1, column=1, padx=5, pady=2)
        self.sagital_label = ttk.Label(nav_frame, text="Slice: 0/0")
        self.sagital_label.grid(row=1, column=2, padx=2)
        
        ttk.Label(nav_frame, text="Coronal:").grid(row=2, column=0, padx=2)
        self.coronal_slider = ttk.Scale(nav_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                       command=lambda v: self.on_slice_change('coronal', v))
        self.coronal_slider.grid(row=2, column=1, padx=5, pady=2)
        self.coronal_label = ttk.Label(nav_frame, text="Slice: 0/0")
        self.coronal_label.grid(row=2, column=2, padx=2)
        
        # Controles de ventana (window level)
        win_frame = ttk.LabelFrame(control_frame, text="Ajuste de Ventana")
        win_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(win_frame, text="Centro:").grid(row=0, column=0, padx=2)
        ttk.Scale(win_frame, from_=-1000, to=3000, variable=self.window_center, 
                 command=self.update_all_views).grid(row=0, column=1, padx=2)
        
        ttk.Label(win_frame, text="Ancho:").grid(row=1, column=0, padx=2)
        ttk.Scale(win_frame, from_=1, to=3000, variable=self.window_width, 
                 command=self.update_all_views).grid(row=1, column=1, padx=2)
        
        # Panel de segmentación avanzada (similar a 3D Slicer)
        seg_frame = ttk.LabelFrame(control_frame, text="Segmentación Avanzada")
        seg_frame.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Threshold range (similar a 3D Slicer)
        ttk.Label(seg_frame, text="Umbral Mín:").grid(row=0, column=0, padx=2)
        tk.Scale(seg_frame, from_=-1000, to=3000, resolution=1, orient=tk.HORIZONTAL,
                variable=self.threshold_min, command=self.update_threshold_display).grid(row=0, column=1, padx=2)
        
        ttk.Label(seg_frame, text="Umbral Máx:").grid(row=1, column=0, padx=2)
        tk.Scale(seg_frame, from_=-1000, to=4000, resolution=1, orient=tk.HORIZONTAL,
                variable=self.threshold_max, command=self.update_threshold_display).grid(row=1, column=1, padx=2)
        
        self.threshold_display = ttk.Label(seg_frame, text="Range: 300-4000")
        self.threshold_display.grid(row=0, column=2, rowspan=2, padx=5)
        
        # Controles adicionales de segmentación
        ttk.Label(seg_frame, text="Vol. Mín:").grid(row=2, column=0, padx=2)
        ttk.Entry(seg_frame, textvariable=self.min_volume_size, width=8).grid(row=2, column=1, padx=2)
        
        ttk.Label(seg_frame, text="Suavizado:").grid(row=3, column=0, padx=2)
        tk.Scale(seg_frame, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL,
                variable=self.smoothing_value).grid(row=3, column=1, padx=2)
        
        ttk.Checkbutton(seg_frame, text="Invertir Volumen", variable=self.invert_volume).grid(row=4, column=0, columnspan=2, pady=2)
        
        # Botones de acción
        ttk.Button(seg_frame, text="Aplicar Threshold", command=self.apply_threshold).grid(row=0, column=3, padx=5)
        ttk.Button(seg_frame, text="Previsualizar 3D", command=self.preview_3d_segmentation).grid(row=1, column=3, padx=5)
        ttk.Button(seg_frame, text="Exportar STL", command=self.export_stl).grid(row=2, column=3, padx=5)
        ttk.Button(seg_frame, text="Reset ROI", command=self.reset_roi).grid(row=3, column=3, padx=5)
        
        # Barra de estado
        self.status_bar = ttk.Label(main_frame, text="Listo", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=5)
    
    def setup_shortcuts(self):
        # Atajos para navegación axial
        self.root.bind("<Left>", lambda e: self.change_slice('axial', -1))
        self.root.bind("<Right>", lambda e: self.change_slice('axial', 1))
        
        # Atajos para navegación sagital (teclas A/D)
        self.root.bind("a", lambda e: self.change_slice('sagital', -1))
        self.root.bind("d", lambda e: self.change_slice('sagital', 1))
        
        # Atajos para navegación coronal (teclas W/S)
        self.root.bind("w", lambda e: self.change_slice('coronal', -1))
        self.root.bind("s", lambda e: self.change_slice('coronal', 1))
        
        # Atajos para window level
        self.root.bind("<Up>", lambda e: self.increase_window())
        self.root.bind("<Down>", lambda e: self.decrease_window())
        
        # Atajos para threshold
        self.root.bind("+", lambda e: self.adjust_threshold(10))
        self.root.bind("-", lambda e: self.adjust_threshold(-10))
    
    def update_threshold_display(self, *args):
        self.threshold_display.config(text=f"Range: {self.threshold_min.get():.0f}-{self.threshold_max.get():.0f}")
    
    def adjust_threshold(self, delta):
        self.threshold_min.set(self.threshold_min.get() + delta)
        self.threshold_max.set(self.threshold_max.get() + delta)
        self.update_threshold_display()
        
    def apply_preset(self, selected):
        vmin, vmax = self.presets[selected]
        if vmin is not None and vmax is not None:
            self.threshold_min.set(vmin)
            self.threshold_max.set(vmax)
            self.update_threshold_display()

    
    def load_dicom_series(self):
        folder = filedialog.askdirectory(title="Seleccionar carpeta DICOM")
        if not folder:
            return
        
        try:
            self.status_bar.config(text="Cargando serie DICOM...")
            self.root.update()
            
            # Cargar usando SimpleITK para manejo profesional de series
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(folder)
            reader.SetFileNames(dicom_files)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            self.sitk_image = reader.Execute()
            
            # Obtener array y orientación correcta
            self.pixel_array = sitk.GetArrayFromImage(self.sitk_image)
            
            # Corregir orientación automáticamente
            self.correct_orientation()
            
            # Configurar navegación para cada vista
            self.current_slices = {
                'axial': self.pixel_array.shape[0] // 2,
                'sagital': self.pixel_array.shape[1] // 2,
                'coronal': self.pixel_array.shape[2] // 2
            }
            
            # Configurar sliders
            self.axial_slider.config(from_=0, to=self.pixel_array.shape[0]-1)
            self.axial_slider.set(self.current_slices['axial'])
            self.axial_label.config(text=f"Slice: {self.current_slices['axial']+1}/{self.pixel_array.shape[0]}")
            
            self.sagital_slider.config(from_=0, to=self.pixel_array.shape[1]-1)
            self.sagital_slider.set(self.current_slices['sagital'])
            self.sagital_label.config(text=f"Slice: {self.current_slices['sagital']+1}/{self.pixel_array.shape[1]}")
            
            self.coronal_slider.config(from_=0, to=self.pixel_array.shape[2]-1)
            self.coronal_slider.set(self.current_slices['coronal'])
            self.coronal_label.config(text=f"Slice: {self.current_slices['coronal']+1}/{self.pixel_array.shape[2]}")
            
            # Inicializar ROI selector
            self.setup_roi_selector()
            
            # Actualizar todas las vistas
            self.update_all_views()
            
            spacing = self.sitk_image.GetSpacing()
            self.status_bar.config(text=f"Serie cargada: {self.pixel_array.shape} | Spacing: {spacing[::-1]} | Orientación {'corregida' if self.orientation_corrected else 'original'}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la serie DICOM:\n{str(e)}")
            self.status_bar.config(text="Error al cargar DICOM")
    
    def correct_orientation(self):
        """Corrige automáticamente la orientación de la imagen"""
        try:
            # Verificar orientación usando la dirección del coseno
            direction = self.sitk_image.GetDirection()
            
            # Si el eje Z está invertido, corregir
            if direction[8] < 0:
                self.pixel_array = np.flip(self.pixel_array, axis=0)
                self.orientation_corrected = True
            
            # Verificar si necesita rotación en los otros ejes
            if direction[0] < 0:
                self.pixel_array = np.flip(self.pixel_array, axis=2)
                self.orientation_corrected = True
                
            if direction[4] < 0:
                self.pixel_array = np.flip(self.pixel_array, axis=1)
                self.orientation_corrected = True
                
        except Exception as e:
            print(f"Error al corregir orientación: {str(e)}")
    
    def setup_roi_selector(self):
        """Configura el selector de ROI en la vista axial"""
        if self.pixel_array is None:
            return
        
        # Mostrar slice axial central inicialmente
        self.update_view('axial')
        
        # Configurar ROI selector
        def onselect(eclick, erelease):
            pass  # La selección se procesa cuando se aplica el threshold
        
        self.roi_selector = RectangleSelector(
            self.ax_axial, onselect, useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True,
            rectprops=dict(facecolor='red', edgecolor='black',
                          alpha=0.2, fill=True))
        
        self.canvas_axial.draw()
    
    def reset_roi(self):
        """Resetea la selección de ROI"""
        if self.roi_selector:
            self.roi_selector.set_active(False)
            self.roi_selector.set_visible(False)
            self.update_view('axial')
            self.status_bar.config(text="ROI resetado")
    
    def apply_threshold(self):
        """Aplica el threshold según los valores seleccionados"""
        if self.pixel_array is None:
            messagebox.showerror("Error", "No hay volumen DICOM cargado")
            return
        
        try:
            self.status_bar.config(text="Aplicando threshold...")
            self.root.update()
            
            # Obtener valores de threshold
            threshold_min = self.threshold_min.get()
            threshold_max = self.threshold_max.get()
            
            # Crear máscara inicial
            mask = (self.pixel_array >= threshold_min) & (self.pixel_array <= threshold_max)
            
            # Aplicar ROI si hay una selección
            if self.roi_selector and self.roi_selector.active:
                # Obtener coordenadas del ROI en píxeles
                x1, x2 = sorted([int(self.roi_selector.extents[0]), int(self.roi_selector.extents[1])])
                y1, y2 = sorted([int(self.roi_selector.extents[2]), int(self.roi_selector.extents[3])])
                
                # Crear máscara de ROI
                roi_mask = np.zeros_like(mask)
                roi_mask[:, y1:y2, x1:x2] = True
                
                # Aplicar ROI a la máscara
                mask = mask & roi_mask
            
            # Eliminar objetos pequeños
            mask = morphology.remove_small_objects(mask, min_size=self.min_volume_size.get())
            
            # Suavizado
            smoothing = self.smoothing_value.get()
            if smoothing > 0:
                mask = morphology.binary_closing(mask, footprint=morphology.ball(smoothing))
                mask = morphology.binary_opening(mask, footprint=morphology.ball(smoothing))
            
            # Invertir si es necesario
            if self.invert_volume.get():
                mask = ~mask
            
            self.current_mask = mask
            self.status_bar.config(text=f"Threshold aplicado: {threshold_min}-{threshold_max} | ROI {'con' if self.roi_selector and self.roi_selector.active else 'sin'} selección")
            
            # Actualizar visualización con la máscara
            self.update_view_with_mask('axial')
            self.update_view_with_mask('sagital')
            self.update_view_with_mask('coronal')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar threshold:\n{str(e)}")
            self.status_bar.config(text="Error en threshold")
    
    def update_all_views(self, event=None):
        if self.pixel_array is None:
            return
        
        self.update_view('axial')
        self.update_view('sagital')
        self.update_view('coronal')
    
    def update_view(self, view_type):
        if self.pixel_array is None:
            return
        
        center = self.window_center.get()
        width = self.window_width.get()
        vmin = center - width/2
        vmax = center + width/2
        
        if view_type == 'axial':
            ax = self.ax_axial
            slice_idx = self.current_slices['axial']
            img = self.pixel_array[slice_idx, :, :]
            title = f"Axial Slice {slice_idx+1}/{self.pixel_array.shape[0]}"
        elif view_type == 'sagital':
            ax = self.ax_sagital
            slice_idx = self.current_slices['sagital']
            img = self.pixel_array[:, slice_idx, :]
            title = f"Sagital Slice {slice_idx+1}/{self.pixel_array.shape[1]}"
        elif view_type == 'coronal':
            ax = self.ax_coronal
            slice_idx = self.current_slices['coronal']
            img = self.pixel_array[:, :, slice_idx]
            title = f"Coronal Slice {slice_idx+1}/{self.pixel_array.shape[2]}"
        else:
            return
        
        ax.clear()
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
        
        if view_type == 'axial':
            self.canvas_axial.draw()
        elif view_type == 'sagital':
            self.canvas_sagital.draw()
        elif view_type == 'coronal':
            self.canvas_coronal.draw()
    
    def update_view_with_mask(self, view_type):
        if self.pixel_array is None or self.current_mask is None:
            return
        
        center = self.window_center.get()
        width = self.window_width.get()
        vmin = center - width/2
        vmax = center + width/2
        
        if view_type == 'axial':
            ax = self.ax_axial
            slice_idx = self.current_slices['axial']
            img = self.pixel_array[slice_idx, :, :]
            mask_slice = self.current_mask[slice_idx, :, :]
            title = f"Axial Slice {slice_idx+1}/{self.pixel_array.shape[0]} (Threshold)"
        elif view_type == 'sagital':
            ax = self.ax_sagital
            slice_idx = self.current_slices['sagital']
            img = self.pixel_array[:, slice_idx, :]
            mask_slice = self.current_mask[:, slice_idx, :]
            title = f"Sagital Slice {slice_idx+1}/{self.pixel_array.shape[1]} (Threshold)"
        elif view_type == 'coronal':
            ax = self.ax_coronal
            slice_idx = self.current_slices['coronal']
            img = self.pixel_array[:, :, slice_idx]
            mask_slice = self.current_mask[:, :, slice_idx]
            title = f"Coronal Slice {slice_idx+1}/{self.pixel_array.shape[2]} (Threshold)"
        else:
            return
        
        ax.clear()
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin)
        
        # Mostrar imagen base
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        
        # Mostrar máscara superpuesta
        mask_slice = mask_slice.astype(float)
        mask_slice[mask_slice == 0] = np.nan  # Hacer transparentes las áreas fuera del threshold
        ax.imshow(mask_slice, cmap='autumn', alpha=0.3, vmin=0, vmax=1)
        
        ax.set_title(title)
        ax.axis('off')
        
        if view_type == 'axial':
            self.canvas_axial.draw()
        elif view_type == 'sagital':
            self.canvas_sagital.draw()
        elif view_type == 'coronal':
            self.canvas_coronal.draw()
    
    def on_slice_change(self, view_type, value):
        if self.pixel_array is None:
            return
        
        slice_idx = int(float(value))
        self.current_slices[view_type] = slice_idx
        
        if view_type == 'axial':
            self.axial_label.config(text=f"Slice: {slice_idx+1}/{self.pixel_array.shape[0]}")
            self.update_view('axial')
            if self.current_mask is not None:
                self.update_view_with_mask('axial')
        elif view_type == 'sagital':
            self.sagital_label.config(text=f"Slice: {slice_idx+1}/{self.pixel_array.shape[1]}")
            self.update_view('sagital')
            if self.current_mask is not None:
                self.update_view_with_mask('sagital')
        elif view_type == 'coronal':
            self.coronal_label.config(text=f"Slice: {slice_idx+1}/{self.pixel_array.shape[2]}")
            self.update_view('coronal')
            if self.current_mask is not None:
                self.update_view_with_mask('coronal')
    
    def change_slice(self, view_type, delta):
        if self.pixel_array is None:
            return
        
        if view_type == 'axial':
            max_slices = self.pixel_array.shape[0]
            slider = self.axial_slider
            label = self.axial_label
        elif view_type == 'sagital':
            max_slices = self.pixel_array.shape[1]
            slider = self.sagital_slider
            label = self.sagital_label
        elif view_type == 'coronal':
            max_slices = self.pixel_array.shape[2]
            slider = self.coronal_slider
            label = self.coronal_label
        else:
            return
        
        new_slice = self.current_slices[view_type] + delta
        new_slice = max(0, min(new_slice, max_slices - 1))
        
        self.current_slices[view_type] = new_slice
        slider.set(new_slice)
        label.config(text=f"Slice: {new_slice+1}/{max_slices}")
        
        self.update_view(view_type)
        if self.current_mask is not None:
            self.update_view_with_mask(view_type)
    
    def increase_window(self):
        self.window_width.set(self.window_width.get() + 50)
        self.update_all_views()
    
    def decrease_window(self):
        new_width = max(10, self.window_width.get() - 50)
        self.window_width.set(new_width)
        self.update_all_views()
    
    def preview_3d_segmentation(self):
        if self.pixel_array is None or self.current_mask is None:
            messagebox.showerror("Error", "Primero aplique un threshold para generar la segmentación")
            return
        
        try:
            self.status_bar.config(text="Generando previsualización 3D...")
            self.root.update()
            
            # Suavizar la máscara para mejor visualización
            smoothed_mask = gaussian_filter(self.current_mask.astype(float), sigma=self.smoothing_value.get())
            
            # Usar marching cubes para superficie 3D
            verts, faces, _, _ = measure.marching_cubes(
                smoothed_mask, 
                level=0.5, 
                step_size=1,  # Mejor resolución
                allow_degenerate=False
            )
            
            # Crear figura 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Mostrar superficie 3D
            mesh = Poly3DCollection(verts[faces], alpha=0.7, edgecolor='k')
            mesh.set_facecolor([0.8, 0.8, 1])
            ax.add_collection3d(mesh)
            
            # Ajustar vista
            ax.set_xlim(0, self.current_mask.shape[2])
            ax.set_ylim(0, self.current_mask.shape[1])
            ax.set_zlim(0, self.current_mask.shape[0])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Previsualización 3D Segmentada')
            
            # Mostrar en ventana nueva
            preview_window = tk.Toplevel(self.root)
            preview_window.title("Previsualización 3D de la Segmentación")
            
            canvas = FigureCanvasTkAgg(fig, master=preview_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, preview_window)
            toolbar.update()
            
            self.status_bar.config(text="Previsualización 3D generada")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en previsualización 3D:\n{str(e)}")
            self.status_bar.config(text="Error en previsualización")
    
    def export_stl(self):
        if self.pixel_array is None or self.current_mask is None:
            messagebox.showerror("Error", "Primero aplique un threshold para generar la segmentación")
            return
        
        try:
            self.status_bar.config(text="Preparando exportación STL...")
            self.root.update()
            
            # Suavizar la máscara para mejor calidad de malla
            smoothed_mask = gaussian_filter(self.current_mask.astype(float), sigma=self.smoothing_value.get())
            
            # Generar malla 3D con marching cubes
            verts, faces, _, _ = measure.marching_cubes(
                smoothed_mask, 
                level=0.5,
                step_size=1,  # Mejor resolución
                allow_degenerate=False
            )
            
            if verts.size == 0 or faces.size == 0:
                messagebox.showerror("Error", "No se pudo generar malla 3D: modelo vacío.")
                self.status_bar.config(text="Error - Malla vacía")
                return
            
            # Crear malla STL
            segmented_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    segmented_mesh.vectors[i][j] = verts[f[j], :]
            
            # Aplicar suavizado adicional a la malla
            segmented_mesh.update_normals()
            
            # Nombre de archivo único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stl_filename = f"segmentacion_3d_{timestamp}.stl"
            
            os.makedirs(self.export_path, exist_ok=True)
            stl_path = os.path.join(self.export_path, stl_filename)
            
            segmented_mesh.save(stl_path)
            
            messagebox.showinfo("Éxito", f"Modelo STL exportado exitosamente:\n{stl_path}")
            self.status_bar.config(text=f"STL exportado: {stl_filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar STL:\n{str(e)}")
            self.status_bar.config(text="Error en exportación STL")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedDICOMViewer(root)
    
    # Menú superior
    menubar = tk.Menu(root)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Abrir serie DICOM", command=app.load_dicom_series)
    filemenu.add_command(label="Configurar ruta de exportación", 
                        command=lambda: app.export_path.set(filedialog.askdirectory()))
    filemenu.add_separator()
    filemenu.add_command(label="Salir", command=root.quit)
    menubar.add_cascade(label="Archivo", menu=filemenu)
    
    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_command(label="Atajos de teclado", 
                        command=lambda: messagebox.showinfo("Atajos", 
                        "Navegación:\n"
                        "Axial: Flechas izquierda/derecha\n"
                        "Sagital: A/D\n"
                        "Coronal: W/S\n"
                        "Window Level: Flechas arriba/abajo\n"
                        "Ajustar threshold: +/-"))
    menubar.add_cascade(label="Ayuda", menu=helpmenu)
    
    root.config(menu=menubar)
    root.mainloop()