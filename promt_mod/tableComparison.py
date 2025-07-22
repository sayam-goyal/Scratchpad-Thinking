import re

def parse_decoded_latent_file(path):
    with open(path, "r") as f:
        text = f.read()
    # Partir el archivo por bloques de pregunta
    bloques = re.split(r'(Question\d+\.\.\.\n)', text)
    questions = []
    # bloques = [ '', 'Question0...', 'contenido', 'Question1...', 'contenido', ...]
    for i in range(1, len(bloques), 2):
        encabezado = bloques[i]
        contenido = bloques[i+1] if (i+1) < len(bloques) else ""
        # Extraer pregunta (primera línea hasta ...), latentes y predicción
        preg_match = re.match(r'Question(\d+)\.\.\.\n(.*?)\.\.\.', encabezado + contenido, re.DOTALL)
        pregunta = preg_match.group(2).strip() if preg_match else "N/A"
        # Buscar todos los latentes
        latents = re.findall(r'decoded \d+th latent \(top5\): (.*)', contenido)
        # Buscar predicción
        pred = "N/A"
        pred_match = re.search(r'Model Prediction: (.*?)<\|endoftext\|>', contenido, re.DOTALL)
        if pred_match:
            pred = pred_match.group(1).strip()
        questions.append({
            "question": pregunta,
            "latents": latents,
            "prediction": pred
        })
    return questions

# Paths to your files
file_normal = "decoded_latent20n.txt" #modify the name of the file
file_changed = "decoded_latent20c.txt" #modify the name of the file
output_file = "latent_table_normal_and_new_promt.md"

normal = parse_decoded_latent_file(file_normal)
changed = parse_decoded_latent_file(file_changed)

assert len(normal) == len(changed), "Los archivos deben tener el mismo número de preguntas."

lines = []
for i, (orig, mod) in enumerate(zip(normal, changed)):
    lines.append(f"\n---\n")
    lines.append(f"### Q {i+1}\n")
    lines.append(f"**Normal:** {orig['question']}")
    lines.append(f"")
    lines.append(f"**New:** {mod['question']}")
    lines.append(f"")
    lines.append(f"| Step | Normal | New |")
    lines.append(f"|------|----------|--------|")
    max_steps = max(len(orig['latents']), len(mod['latents']))
    for j in range(max_steps):
        lat_o = orig['latents'][j] if j < len(orig['latents']) else ""
        lat_m = mod['latents'][j] if j < len(mod['latents']) else ""
        lines.append(f"| {j}th | {lat_o} | {lat_m} |")
    lines.append(f"| pred | {orig['prediction']} | {mod['prediction']} |")
    lines.append("")  # Doble salto para separación visual

with open(output_file, "w") as f:
    f.write("\n".join(lines))

