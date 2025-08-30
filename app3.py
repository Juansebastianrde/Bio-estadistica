# Streamlit: ejecuta un .ipynb "celda por celda" directamente desde GitHub (1:1)
# Tu URL ya está preconfigurada más abajo en DEFAULT_GH_URL.

import streamlit as st
st.set_page_config(page_title="Notebook", layout="wide")
st.title("Notebook proyecto ML bioestadistica")

import io, re, ast, requests, os
from datetime import datetime
from pathlib import Path
import nbformat
import pandas as pd
import matplotlib.pyplot as plt

# ===== Asegurar que el CWD sea la carpeta del script =====
BASE = Path(__file__).parent.resolve()
try:
    os.chdir(BASE)  # <-- con esto, pd.read_csv("HDHI Admission data.csv") busca en Downloads
except Exception as _:
    pass
st.caption(f"Working dir: {os.getcwd()}")

# ===== Compat shims: print/display/plt.show/plotly.show =====
print = st.write  # noqa: F811

def display(*objs):
    for obj in objs:
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                st.dataframe(obj, use_container_width=True)
            else:
                st.write(obj)
        except Exception:
            st.text(repr(obj))

class Markdown(str): ...
class HTML(str): ...

def _plt_show(*args, **kwargs):
    st.pyplot(plt.gcf(), clear_figure=False)
plt.show = _plt_show

try:
    import plotly.graph_objects as go
    def _plotly_show(self, *_, **__):
        st.plotly_chart(self, use_container_width=True)
    go.Figure.show = _plotly_show
except Exception:
    pass

# ===== Utilidades =====
def sanitize_code(code: str) -> str:
    out = []
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("%") or s.startswith("!"):
            out.append(f"# [omitido del notebook] {ln}")
        elif "get_ipython()" in ln:
            out.append(f"# [omitido get_ipython] {ln}")
        else:
            out.append(ln)
    return "\n".join(out)

def exec_cell(code: str, ns: dict, show_last_expr: bool = True):
    clean = sanitize_code(code)
    if not show_last_expr:
        exec(clean, ns)
        return
    try:
        tree = ast.parse(clean, mode="exec")
        if not tree.body:
            return
        last_is_expr = isinstance(tree.body[-1], ast.Expr)
        if last_is_expr:
            body = ast.Module(body=tree.body[:-1], type_ignores=[])
            last = ast.Expression(body=tree.body[-1].value)
            exec(compile(body, "<cell>", "exec"), ns)
            val = eval(compile(last, "<cell>", "eval"), ns)
            if val is not None:
                display(val)
        else:
            exec(clean, ns)
    except Exception:
        exec(clean, ns)

def to_raw_github_url(url: str) -> str | None:
    url = url.strip()
    if not url:
        return None
    if "raw.githubusercontent.com" in url:
        return url
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    m2 = re.match(r"https://gist\.github\.com/([^/]+)/([a-f0-9]+)", url)
    if m2:
        if "/raw" in url:
            return url.replace("gist.github.com", "gist.githubusercontent.com")
        return url.replace("gist.github.com", "gist.githubusercontent.com") + "/raw"
    return None

def fetch_ipynb_from_github(url: str, token: str | None = None) -> bytes:
    raw_url = to_raw_github_url(url)
    if not raw_url:
        raise ValueError("URL de GitHub inválida para convertir a raw.")
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    r = requests.get(raw_url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Descarga fallida ({r.status_code}). Revisa URL/permiso.")
    return r.content

# ===== Sidebar (opciones) =====
st.sidebar.header("Fuente del notebook")
DEFAULT_GH_URL = "https://github.com/Juansebastianrde/Reduccion-de-dimensionalidad/blob/main/Proyecto_ML%20(1).ipynb"
gh_url = st.sidebar.text_input("URL de GitHub (.ipynb, blob o raw)", value=DEFAULT_GH_URL)
gh_token = st.sidebar.text_input("Token GitHub (opcional, para repos privados)", type="password")
uploaded = st.sidebar.file_uploader("…o sube tu .ipynb", type=["ipynb"])
btn_reload = st.sidebar.button("Cargar")

stop_on_error = st.sidebar.checkbox("Detener al primer error", value=False)
show_code = st.sidebar.checkbox("Mostrar código de cada celda", value=True)

# ===== Obtener bytes del notebook =====
nb_bytes = None
nb_label = None

try:
    if uploaded is not None:
        nb_bytes = uploaded.read()
        nb_label = uploaded.name
    else:
        # Auto-cargar desde la URL (o cuando pulses "Cargar")
        if btn_reload or True:  # True -> carga en arranque
            nb_bytes = fetch_ipynb_from_github(gh_url, gh_token or None)
            nb_label = gh_url
except Exception as e:
    st.error(f"Error al obtener el notebook: {e}")
    st.stop()

# ===== Leer y ejecutar =====
try:
    nb = nbformat.read(io.BytesIO(nb_bytes), as_version=4)
except Exception as e:
    st.error(f"No pude leer el .ipynb: {e}")
    st.stop()

st.caption(f"Notebook cargado: **{nb_label}** · {len(nb.cells)} celdas · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Namespace compartido (como en Jupyter)
ns = {
    "__name__": "__main__",
    "st": st,
    "pd": pd,
    "plt": plt,
    "display": display,
    "Markdown": Markdown,
    "HTML": HTML,
    "print": print,
    # Exponer BASE para que celdas puedan usarla si quieren
    "BASE": BASE,
    "Path": Path,
}

try:
    import numpy as np
    ns["np"] = np
except Exception:
    pass

error_found = False
for i, cell in enumerate(nb.cells, start=1):
    ctype = cell.get("cell_type")
    if ctype == "markdown":
        st.markdown(cell.get("source", ""), unsafe_allow_html=True)
        st.divider()
        continue
    if ctype != "code":
        continue

    code = cell.get("source", "") or ""
    if not code.strip():
        continue

    with st.container(border=True):
        st.caption(f"Code cell {i}")
        if show_code:
            st.code(code, language="python")
        try:
            exec_cell(code, ns, show_last_expr=True)
        except Exception as e:
            error_found = True
            st.exception(e)
            if stop_on_error:
                st.error(f"Detenido por error en la celda {i}. Desactiva 'Detener al primer error' para continuar.")
                break

st.success("Ejecución completada." if not error_found else "Ejecución finalizada con errores (revisa las celdas marcadas).")
