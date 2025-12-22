import os
import shutil
import re 

source_dir = '/home/pedro-furlan/Downloads/dataset_breast'
dest_dir = '/home/pedro-furlan/Área de trabalho/codigos/dataset'

valid_views = ['CC', 'MLO']
extensions = ['.png', '.jpg', '.jpeg', '.dcm']
exclude_keyword = 'mask'

print(f"Iniciando varredura em: {source_dir}")
print(f"Excluindo arquivos que contêm: '{exclude_keyword}'")

copied_files = 0
ignored_files = 0

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.startswith('.'):
            continue

        file_lower = file.lower()
        original_path = os.path.join(root, file)

        if exclude_keyword in file_lower:
            print(f"Ignorando (mask): {original_path}")
            ignored_files += 1
            continue
        
        has_extension = any(file_lower.endswith(ext) for ext in extensions)
        if not has_extension:
            continue

        filename_no_ext = os.path.splitext(file)[0]
        parts = re.split(r'[._-]', filename_no_ext.upper())
        has_valid_view = any(view in parts for view in valid_views)

        if has_valid_view:
            relative_path = os.path.relpath(root, source_dir)
            new_dest_dir = os.path.join(dest_dir, relative_path)
            
            os.makedirs(new_dest_dir, exist_ok=True)
            
            shutil.copy2(original_path, os.path.join(new_dest_dir, file))
            
            print(f"Copiando: {original_path}")
            copied_files += 1

print(f"--- CONCLUÍDO ---")
print(f"Total de arquivos copiados: {copied_files}")
print(f"Total de arquivos ignorados (mask): {ignored_files}")
print(f"Verifique a pasta: {dest_dir}")