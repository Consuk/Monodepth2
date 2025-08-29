import os
import argparse
import numpy as np
import PIL.Image as pil
import cv2
import glob

from utils import readlines

def try_candidates(root_dir, folder, file_id):
    """
    Intenta encontrar el archivo de profundidad probando varias combinaciones:
    - Conservar folder completo (datasetX/keyframeY)
    - Usar solo datasetX
    - Variantes de nombre: 390.png, 000390.png, 0000390.png
    - Extensiones: .png, .tif, .tiff
    """
    # candidatos de nombres (sin y con cero-padding)
    ids = [str(file_id)]
    try:
        n = int(file_id)
        ids.extend([f"{n:04d}", f"{n:05d}", f"{n:06d}"])
    except ValueError:
        # si no es numérico, dejamos solo el string original
        pass

    names = []
    for base in ids:
        names.append(f"{base}.png")
        names.append(f"{base}.tif")
        names.append(f"{base}.tiff")

    # candidatos de carpetas
    # 1) conservar datasetX/keyframeY
    cand_dirs = [
        os.path.join(root_dir, folder, "Ground_truth_CT", "DepthL"),
        # 2) solo datasetX (por si la verdad está un nivel arriba)
        os.path.join(root_dir, folder.split("/")[0], "Ground_truth_CT", "DepthL"),
    ]

    # 3) búsqueda recursiva como último recurso
    for cand_dir in cand_dirs:
        for name in names:
            p = os.path.join(cand_dir, name)
            if os.path.exists(p):
                return p

    # fallback: glob recursivo dentro de root_dir por cualquier match *file_id*.(png|tif|tiff)
    for ext in ("png", "tif", "tiff"):
        hits = glob.glob(os.path.join(root_dir, "**", f"*{file_id}*.{ext}"), recursive=True)
        if hits:
            # elige el primero determinísticamente (orden alfabético)
            hits.sort()
            return hits[0]

    return None


def export_gt_depths_kitti():
    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path', type=str, required=True,
                        help='path to the root of the KITTI/EndoVIS/SCaRED data')
    parser.add_argument('--split', type=str, required=True,
                        choices=["eigen", "eigen_benchmark", "hamlyn", "SERV-CT", "endovis"],
                        help='which split to export gt from')
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print(f"Exporting ground truth depths for {opt.split}")

    # Ajusta aquí el “root” real donde viven las GT de este split.
    # En tu ejemplo estaban debajo de .../SERV-CT/
    gt_root = os.path.join(opt.data_path, "SERV-CT")

    gt_depths = []
    missing = 0

    for line in lines:
        folder, file_id, _ = line.split()  # p.ej. folder="dataset3/keyframe4", file_id="390"
        print(line.strip())

        # No descartes "keyframe4": úsalo completo
        # Busca el archivo probando múltiples variantes
        gt_depth_path = try_candidates(gt_root, folder, file_id)
        print("→ candidato:", gt_depth_path if gt_depth_path else "(no encontrado)")

        if gt_depth_path is None or not os.path.exists(gt_depth_path):
            print("⚠️  Archivo de GT no encontrado. Revisa estructura/carpetas.")
            missing += 1
            continue

        # Lee preservando profundidad real: si es PNG 16-bit, IMREAD_UNCHANGED devuelve uint16
        im = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"⚠️  No se pudo leer con OpenCV: {gt_depth_path}")
            missing += 1
            continue

        # Si llega como RGB, convertir a gris; si ya viene single-channel, lo dejamos
        if im.ndim == 3:
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            im_gray = im

        # Normaliza: muchos GT depth PNG usan 16-bit con factor 256
        # (ajusta si tu dataset usa otra escala)
        if im_gray.dtype == np.uint16:
            gt_depth = im_gray.astype(np.float32) / 256.0
        else:
            # si llegó 8-bit, intenta la misma escala; ajusta según tu dataset
            gt_depth = im_gray.astype(np.float32) / 256.0

        print("shape:", gt_depth.shape, "dtype:", gt_depth.dtype, "min/max:", float(gt_depth.min()), float(gt_depth.max()))
        gt_depths.append(gt_depth)

    if not gt_depths:
        raise RuntimeError("No se pudo cargar ningún GT. Verifica rutas y nombres en 'test_files.txt' y estructura de carpetas.")

    output_path = os.path.join(split_folder, "gt_depths.npz")
    print(f"Saving to {opt.split} {output_path}")
    np.savez_compressed(output_path, data=np.array(gt_depths))

    if missing:
        print(f"Nota: {missing} archivos GT no se encontraron. Revisa esos casos.")


if __name__ == "__main__":
    export_gt_depths_kitti()
