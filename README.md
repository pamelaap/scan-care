🧠 Scan & Care - Plataforma de Segmentación Médica y Exportación 3D
Bienvenid@ a Scan & Care, una herramienta intuitiva y gratuita que permite a profesionales de la salud transformar estudios médicos DICOM en modelos 3D imprimibles. Nuestra misión: acercar la ingeniería médica al usuario clínico.

🚀 ¿Qué puedes hacer con esta app?
Cargar carpetas de estudios médicos (DICOM)
Visualizar imágenes en 3 planos: axial, sagital y coronal
Seleccionar estructuras a segmentar (hueso, tumor, tejido blando…)
Aplicar umbrales automáticos o manuales
Generar un modelo 3D a partir del área segmentada
Exportar el modelo en formato .STL listo para impresión 3D
🛠️ Requisitos
Python 3.9 o superior
Pip (gestor de paquetes de Python)
📦 Instalación paso a paso
1. Clona este repositorio
git clone https://github.com/tuusuario/scan-and-care.git cd scan-and-care

python -m venv venv source venv/bin/activate # En Windows usa: venv\Scripts\activate

pip install -r requirements.txt

python scan_and_care_preset_threshold.py

🧪 ¿Cómo se usa? 📁 Paso 1: Carga de imágenes médicas Haz clic en Archivo > Abrir serie DICOM y selecciona una carpeta (no archivo) con imágenes DICOM.

🧠 Paso 2: Visualiza el estudio Navega por los tres planos: axial, coronal, sagital.

🎯 Paso 3: Segmentación Selecciona una estructura:

Manual

Hueso (300–3000 HU)

Tumor cerebral (40–80 HU)

Tejido blando (-100–300 HU)

Haz clic en "Aplicar threshold".

🧾 Paso 4: Previsualización 3D Presiona "Previsualizar 3D" para generar una vista en 3D del modelo.

📤 Paso 5: Exportación Haz clic en "Exportar STL". El archivo se guardará en la carpeta Descargas/DICOM_Exports.

🎮 Atajos útiles Ejes:

⬅➡ (izquierda/derecha): plano axial

A/D: plano sagital

W/S: plano coronal

Nivel de brillo/contraste: ⬆⬇

Threshold: + / -

🖨️ ¿Cómo imprimo el modelo STL? Puedes importar el .stl a cualquier software de slicing como:

Ultimaker Cura

PrusaSlicer

Lychee Slicer (resina)

📚 Recursos útiles 🧾 Guías sobre materiales de impresión: PLA vs ABS vs PETG

Guía oficial de materiales para impresión médica - Stratasys

Consideraciones clínicas de impresión 3D en medicina

📹 Tutoriales: Importar STL a Cura (YouTube)

Impresión de modelos médicos paso a paso

👩‍🔬 Créditos Proyecto desarrollado como parte del reto de Procesamiento de Imágenes Médicas - Ingeniería Biomédica, Tecnológico de Monterrey.
