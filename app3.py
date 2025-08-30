import streamlit as st
import nbformat
from nbclient import NotebookClient
import tempfile
import os
import contextlib
import io
import base64
import json
from typing import Dict, Any, Set, List
from pathlib import Path

# =========================
# Config de página
# =========================
st.set_page_config(page_title="Refactor 1:1 de Notebook → Módulo + App", page_icon="📓", layout="wide")
st.title("📓 ➜ 🧩 Refactor 1:1: Notebook a módulo Python + ejecución con outputs idénticos")

# =========================
# Utilidades
# =========================
@contextlib.contextmanager
def chdir(path: str):
    prev = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def _b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64.encode("utf-8"))

def _render_output_item(data: dict):
    """Renderiza un dict de datos MIME de un output de nbclient."""
    # Imágenes
    if "image/png" in data:
        st.image(io.BytesIO(_b64_to_bytes(data["image/png"])))
        return True
    if "image/jpeg" in data:
        st.image(io.BytesIO(_b64_to_bytes(data["image/jpeg"])))
        return True

    # HTML enriquecido
    if "text/html" in data:
        from streamlit.components.v1 import html as st_html
        st_html(data["text/html"], height=420, scrolling=True)
        return True

    # LaTeX
    if "text/latex" in data:
        st.latex(data["text/latex"])
        return True

    # JSON
    if "application/json" in data:
        st.json(data["application/json"])
        return True

    # Vega/Vega-Lite (Altair)
    for k in list(data.keys()):
        if "vnd.vegalite" in k or "vnd.vega" in k:
            try:
                st.vega_lite_chart(data[k], use_container_width=True)
                return True
            except Exception:
                pass

    # Plotly
    for k in ["application/vnd.plotly.v1+json", "application/vnd.plotly.v2+json"]:
        if k in data:
            try:
                import plotly.io as pio
                fig = pio.from_json(data[k] if isinstance(data[k], str) else io.StringIO(data[k]))
                st.plotly_chart(fig, use_container_width=True)
                return True
            except Exception:
                st.json(data[k])
                return True

    # Texto plano (repr)
    if "text/plain" in data:
        st.code(str(data["text/plain"]), language="text")
        return True

    return False

def _render_cell_outputs(cell, show_code: bool, idx: int):
    """Pinta una celda ejecutada (código y salidas) con nbclient."""
    if show_code and cell.get("source"):
        with st.expander(f"🧩 Celda {idx} — código", expanded=False):
            st.code(cell["source"], language="python")

    outputs = cell.get("outputs", []) or []
    if not outputs:
        return

    for out in outputs:
        ot = out.get("output_type")
        if ot == "stream":
            text = out.get("text", "")
            name = out.get("name", "stdout")
            if text.strip():
                if name == "stderr":
                    st.warning(text)
                else:
                    st.code(text, language="text")
        elif ot == "error":
            ename = out.get("ename", "Error")
            evalue = out.get("evalue", "")
            st.error(f"{ename}: {evalue}")
            trace = out.get("traceback", [])
            if trace:
                st.code("\n".join(trace), language="text")
        elif ot in ("display_data", "execute_result"):
            data = out.get("data", {})
            rendered = _render_output_item(data)
            if not rendered:
                st.caption(f"Salida no reconocida. Claves MIME: {list(data.keys())}")
        else:
            st.caption(f"(Salida '{ot}' no estándar)")

def _inject_parameters(nb, params: Dict[str, Any]):
    """Inserta/reemplaza parámetros: celda con tag 'parameters' o se agrega al inicio."""
    if not params:
        return nb
    params_json = json.dumps(params, ensure_ascii=False)
    lines = [
        "import json as _json",
        f"PARAMS = _json.loads(r'''{params_json}''')",
        "# Variables de conveniencia (si existen en PARAMS):",
    ]
    for k in params.keys():
        if isinstance(k, str) and k.isidentifier():
            lines.append(f"{k} = PARAMS.get('{k}')")
    cell_src = "\n".join(lines)
    param_cell = nbformat.v4.new_code_cell(source=cell_src, metadata={"tags": ["parameters"]})

    # Buscar celda 'parameters' existente
    idx = None
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") == "code" and "tags" in c.get("metadata", {}) and "parameters" in c["metadata"]["tags"]:
            idx = i
            break
    if idx is not None:
        nb.cells[idx] = param_cell
    else:
        nb.cells.insert(0, param_cell)
    return nb

def _apply_skip_tags(nb, tags_to_skip: Set[str]) -> int:
    """Reemplaza el código de celdas con ciertos tags por un no-op."""
    if not tags_to_skip:
        return 0
    count = 0
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        tags = set(c.get("metadata", {}).get("tags", []))
        if tags & tags_to_skip:
            c["source"] = f"print('⏭️ Celda saltada por tags: {', '.join(sorted(tags & tags_to_skip))}')"
            count += 1
    return count

def _first_line_preview(src: str, max_len: int = 80) -> str:
    s = (src or "").strip().splitlines()[:1]
    line = s[0] if s else ""
    line = line.replace("\t", " ").strip()
    return (line[: max_len - 1] + "…") if len(line) > max_len else line

def build_refactored_module(nb, module_name: str = "nb_pipeline") -> bytes:
    """
    Construye un módulo Python con clase Pipeline y una función por celda de código.
    - Las celdas se conservan en orden.
    - Estado compartido en self.ns
    - Soporta set_params(dict) (crea PARAMS y variables sueltas)
    - Cada celda llama a _exec_b64(b64_code, self.ns)
    """
    header = f'''# Auto-generado desde notebook (refactor 1:1)
# Módulo: {module_name}.py
# Conserva el orden y el estado entre celdas.
from typing import Dict, Any, Iterable, Optional
import base64

def _exec_b64(b64: str, ns: dict):
    code = base64.b64decode(b64.encode("utf-8")).decode("utf-8")
    exec(code, ns)

class Pipeline:
    def __init__(self):
        self.ns: Dict[str, Any] = {{}}

    def set_params(self, params: Dict[str, Any]):
        self.ns["PARAMS"] = dict(params or {{}})
        # Variables de conveniencia
        for k, v in list(self.ns["PARAMS"].items()):
            if isinstance(k, str) and k.isidentifier():
                self.ns[k] = v

'''
    body = []
    cell_idx = 0
    for i, c in enumerate(nb.cells, start=1):
        if c.get("cell_type") != "code":
            # Conservar markdown como comentarios para referencia
            md = (c.get("source") or "").splitlines()
            if md:
                body.append("\n" + "\n".join([f"# [md] {line}" for line in md]))
            continue

        cell_idx += 1
        tags = c.get("metadata", {}).get("tags", [])
        code_src = c.get("source") or ""
        # Base64 para embebido robusto
        b64 = base64.b64encode(code_src.encode("utf-8")).decode("utf-8")
        preview = _first_line_preview(code_src, 70).replace("'", "\\'")
        tag_comment = f"  # tags: {','.join(tags)}" if tags else ""
        fn = f"""    def cell_{cell_idx:03d}(self):
        \"\"\"Celda {cell_idx} — {preview}{tag_comment}\"\"\"
        _exec_b64('{b64}', self.ns)
"""
        body.append(fn)

    # run(all) y run_until
    footer = f"""
    def run_all(self):
        \"\"\"Ejecuta todas las celdas en orden.\"\"\"
        for name in dir(self):
            if name.startswith("cell_"):
                getattr(self, name)()

    def run_until(self, last_cell: int):
        \"\"\"Ejecuta desde el inicio hasta 'last_cell' (incluido).\"\"\"
        for i in range(1, {cell_idx}+1):
            getattr(self, f"cell_{{i:03d}}")()
            if i >= last_cell:
                break
"""
    module_text = header + "\n".join(body) + footer
    return module_text.encode("utf-8")

# =========================
# Sidebar (Controles)
# =========================
st.sidebar.header("⚙️ Configuración")
timeout = st.sidebar.slider("Timeout por ejecución (segundos)", 30, 3600, 600, 10)
show_code = st.sidebar.checkbox("Mostrar código de las celdas", value=False)
kernel = st.sidebar.text_input("Kernel de Python", value="python3", help="Normalmente 'python3'")

st.sidebar.subheader("Parámetros (opcional)")
default_params = """{
  "alpha": 0.2,
  "n_rows": 500,
  "umbral": 0.5
}"""
params_text = st.sidebar.text_area("JSON de parámetros (se inyectan en PARAMS y como variables)", value=default_params, height=160)

st.sidebar.subheader("Saltar celdas por tag (opcional)")
skip_tags_text = st.sidebar.text_input("Tags separados por coma (ej: slow,skip)", value="")

st.sidebar.subheader("Archivos auxiliares (opcional)")
aux_files = st.sidebar.file_uploader(
    "Sube archivos que tu notebook usa por ruta relativa (CSV/Parquet/imágenes, etc.)",
    accept_multiple_files=True
)

# =========================
# Input Notebook
# =========================
uploaded_nb = st.file_uploader("📥 Sube tu Notebook (.ipynb)", type=["ipynb"])

if uploaded_nb is None:
    st.info("Sube tu archivo .ipynb para empezar (generaremos un módulo Python 1:1 y podrás ejecutarlo con outputs idénticos).")
    st.stop()

# =========================
# Parse / Resumen
# =========================
try:
    raw_nb = nbformat.reads(uploaded_nb.getvalue().decode("utf-8"), as_version=4)
except Exception as e:
    st.error(f"No se pudo leer el notebook: {e}")
    st.stop()

total_cells = len(raw_nb.cells)
md_cells = sum(1 for c in raw_nb.cells if c.get("cell_type") == "markdown")
code_cells = sum(1 for c in raw_nb.cells if c.get("cell_type") == "code")
st.success(f"Notebook cargado: {total_cells} celdas ({code_cells} de código, {md_cells} markdown) ✔️")

with st.expander("🧭 Mapa de celdas de código"):
    rows = []
    idx = 0
    for c in raw_nb.cells:
        if c.get("cell_type") != "code":
            continue
        idx += 1
        tags = c.get("metadata", {}).get("tags", [])
        rows.append(f"{idx:03d} | {(_first_line_preview(c.get('source') or ''))} " + (f"[tags: {','.join(tags)}]" if tags else ""))
    if rows:
        st.text("\n".join(rows))
    else:
        st.caption("No se detectaron celdas de código.")

# =========================
# Ejecución con nbclient (outputs idénticos)
# =========================
run = st.button("▶️ Ejecutar Notebook (outputs idénticos)", type="primary", use_container_width=True)

if run:
    # Validar parámetros
    try:
        params = json.loads(params_text) if params_text.strip() else {}
        if not isinstance(params, dict):
            st.warning("Los parámetros deben ser un JSON de objeto (dict). Ignorando…")
            params = {}
    except Exception as e:
        st.error(f"JSON de parámetros inválido: {e}")
        params = {}

    # Tags a saltar
    tags_to_skip: Set[str] = set()
    if skip_tags_text.strip():
        tags_to_skip = {t.strip() for t in skip_tags_text.split(",") if t.strip()}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Guardar archivos auxiliares con su nombre original (para rutas relativas)
        if aux_files:
            for f in aux_files:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as fh:
                    fh.write(f.getbuffer())

        # Clonar el notebook para modificaciones en memoria
        nb = nbformat.from_dict(raw_nb)
        if params:
            nb = _inject_parameters(nb, params)
        skipped = _apply_skip_tags(nb, tags_to_skip)

        try:
            with st.spinner(f"Ejecutando notebook… (timeout: {timeout}s)"):
                client = NotebookClient(
                    nb,
                    timeout=timeout,
                    kernel_name=kernel,
                    allow_errors=True  # seguimos para mostrar todas las salidas
                )
                with chdir(tmpdir):
                    executed = client.execute()

            st.success(f"Ejecución completa ✔️ (celdas saltadas: {skipped})")

            # Render: markdown + salidas de celdas de código
            for i, cell in enumerate(executed.cells, start=1):
                ctype = cell.get("cell_type")
                if ctype == "markdown":
                    st.markdown(cell.get("source", ""))
                elif ctype == "code":
                    _render_cell_outputs(cell, show_code=show_code, idx=i)

            # Descargar notebook ejecutado
            out_bytes = nbformat.writes(executed).encode("utf-8")
            st.download_button(
                "⬇️ Descargar notebook ejecutado (.ipynb)",
                data=out_bytes,
                file_name="notebook_ejecutado.ipynb",
                mime="application/x-ipynb+json"
            )

        except Exception as e:
            st.error(f"Fallo ejecutando el notebook: {e}")

st.markdown("---")

# =========================
# Refactor 1:1 → Módulo Python
# =========================
st.subheader("🧩 Descargar módulo refactorizado 1:1 (Python)")
mod_name = st.text_input("Nombre del módulo (sin .py)", value="nb_pipeline")

try:
    module_bytes = build_refactored_module(raw_nb, module_name=mod_name)
    st.download_button(
        "⬇️ Descargar módulo refactorizado",
        data=module_bytes,
        file_name=f"{mod_name}.py",
        mime="text/x-python"
    )
    with st.expander("Ver vista previa del módulo"):
        st.code(module_bytes.decode("utf-8")[:4000] + ("\n# ... (truncado)" if len(module_bytes) > 4000 else ""), language="python")
except Exception as e:
    st.error(f"No se pudo construir el módulo: {e}")

st.markdown("""
**Uso del módulo generado**:

```python
from nb_pipeline import Pipeline  # o el nombre que elegiste

pipe = Pipeline()
pipe.set_params({"alpha": 0.2, "n_rows": 500})  # opcional
pipe.cell_001()       # ejecutar celda 1
pipe.run_until(5)     # o ejecutar hasta la 5
pipe.run_all()        # o todo el pipeline
# El estado vivo está en pipe.ns (diccionario con variables)
""")

