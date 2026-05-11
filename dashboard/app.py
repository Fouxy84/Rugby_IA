"""
Dashboard Rugby IA — Streamlit
Interface temps réel pour l'analyse de matchs de rugby.

Sections :
  🏉 Sources       — Upload / téléchargement / recherche de matchs
  📊 Analyse       — Vidéo annotée + phase de jeu en cours
  🗺️ Heatmaps      — Cartes de chaleur par équipe et ballon
  ⚡ Événements    — Timeline des événements clés
  🧩 Patterns      — Patterns tactiques détectés
  📈 Insights      — Key Insights automatiques
  📋 Stats match   — Tableau de bord global
"""

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import websockets
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CFG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f)

API_BASE = f"http://localhost:{CFG['api']['port']}"
WS_BASE  = f"ws://localhost:{CFG['api']['port']}"

st.set_page_config(
    page_title="Rugby IA",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personnalisé
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a472a;
        text-align: center;
        padding: 1rem 0;
    }
    .phase-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #1a472a, #2d6a2d);
        color: white;
    }
    .event-card {
        border-left: 4px solid #e74c3c;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        background: #ffeaea;
        border-radius: 4px;
    }
    .insight-card {
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        background: linear-gradient(90deg, #e8f5e9, #f1f8e9);
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        font-size: 0.95rem;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        background: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "match_id": None,
        "match_name": "",
        "is_analyzing": False,
        "snapshot_history": [],
        "events": [],
        "patterns": [],
        "current_phase": "—",
        "phase_conf": 0.0,
        "insights": [],
        "zone_stats": {},
        "n_home": 0,
        "n_away": 0,
        "processing_fps": 0.0,
        "heatmap_mode": "global",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Helpers API
# ---------------------------------------------------------------------------

def api_get(path: str, **params) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error : {e}")
        return None


def api_post(path: str, **kwargs) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error : {e}")
        return None


def check_api() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sidebar — Navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🏉 Rugby IA")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📥 Sources", "▶️ Analyse temps réel", "🗺️ Heatmaps",
     "⚡ Événements", "🧩 Patterns", "📈 Insights", "📋 Statistiques"],
    label_visibility="collapsed",
)

# Status API
api_ok = check_api()
if api_ok:
    st.sidebar.success("✅ API connectée")
else:
    st.sidebar.error("❌ API hors ligne")
    st.sidebar.info("Lancez l'API : `uvicorn src.api.main:app --reload`")

if st.session_state.match_id:
    st.sidebar.markdown(f"**Match actif :** `{st.session_state.match_name}`")
    st.sidebar.markdown(f"**ID :** `{st.session_state.match_id}`")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Rugby IA v1.0 — DataScientest MLOps</small>",
    unsafe_allow_html=True,
)


# ===========================================================================
# PAGE : Accueil
# ===========================================================================

if page == "🏠 Accueil":
    st.markdown('<div class="main-header">🏉 Rugby IA — Analyse Intelligente de Matchs</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Bienvenue sur le tableau de bord **Rugby IA**.  
    Ce système analyse automatiquement les matchs de rugby par intelligence artificielle.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Détection</h3>
            <p>Joueurs, arbitres et ballon identifiés en temps réel via YOLOv8</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📍 Phases</h3>
            <p>Mêlée, touche, essai, ruck, maul… reconnus automatiquement</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🗺️ Heatmaps</h3>
            <p>Visualisation de l'utilisation de l'espace par équipe</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🧩 Patterns</h3>
            <p>Switch, linebreak, pick & go, défense rideau…</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Démarrage rapide")
    st.markdown("""
    1. **Aller dans 📥 Sources** pour importer un match (upload ou URL YouTube)
    2. **Cliquer sur ▶️ Analyse temps réel** pour lancer l'analyse
    3. **Explorer** les heatmaps, événements et insights
    """)

    if not api_ok:
        st.warning("⚠️ L'API n'est pas démarrée. Lancez-la avec :")
        st.code("uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")


# ===========================================================================
# PAGE : Sources
# ===========================================================================

elif page == "📥 Sources":
    st.header("📥 Sources de matchs")

    tab_upload, tab_url, tab_search, tab_leagues = st.tabs(
        ["📁 Upload local", "🔗 Télécharger URL", "🔍 Rechercher", "🏆 Compétitions"]
    )

    # --- Upload ---
    with tab_upload:
        st.subheader("Importer un fichier vidéo")
        st.info(f"Formats acceptés : {', '.join(CFG['video']['supported_formats'])} — Max {CFG['api']['max_upload_size_mb']} MB")

        uploaded = st.file_uploader(
            "Choisir un fichier vidéo",
            type=CFG["video"]["supported_formats"],
        )
        match_name = st.text_input("Nom du match (optionnel)")

        if st.button("⬆️ Importer", disabled=not uploaded or not api_ok):
            with st.spinner("Upload en cours…"):
                files = {"file": (uploaded.name, uploaded.getvalue())}
                data = {"match_name": match_name or uploaded.name}
                resp = api_post("/api/video/upload", files=files, data=data)
                if resp:
                    st.session_state.match_id = resp["match_id"]
                    st.session_state.match_name = match_name or uploaded.name
                    st.success(f"✅ Match importé : ID `{resp['match_id']}` ({resp['size_mb']} MB)")

    # --- URL ---
    with tab_url:
        st.subheader("Télécharger depuis YouTube / Dailymotion / Vimeo")
        url = st.text_input("URL de la vidéo")
        fn = st.text_input("Nom du fichier (optionnel, sans extension)")

        if st.button("⬇️ Télécharger", disabled=not url or not api_ok):
            with st.spinner("Téléchargement en cours (peut prendre quelques minutes)…"):
                resp = api_post("/api/video/download", json={"url": url, "filename": fn or None})
                if resp:
                    st.session_state.match_id = resp["match_id"]
                    st.session_state.match_name = fn or resp["filename"]
                    st.success(f"✅ Téléchargé : `{resp['filename']}` — ID `{resp['match_id']}`")

    # --- Recherche ---
    with tab_search:
        st.subheader("Rechercher un match sur YouTube")
        query = st.text_input("Recherche", placeholder="Top14 2024 Toulouse Racing highlights")
        if st.button("🔍 Rechercher", disabled=not api_ok):
            with st.spinner("Recherche…"):
                data = api_get("/api/video/search", q=query)
                if data:
                    for r in data.get("results", []):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f"**{r['title']}**")
                            st.caption(f"Durée : {r.get('duration', '?')}s — Vues : {r.get('view_count', '?'):,}")
                        with col_b:
                            if st.button("⬇️ Télécharger", key=r["url"]):
                                st.session_state._dl_url = r["url"]
                                st.info(f"URL copiée dans l'onglet 'Télécharger URL' : {r['url']}")

    # --- Ligues ---
    with tab_leagues:
        st.subheader("Matchs récents par compétition")
        league = st.selectbox("Compétition", list(RugbyDataConnector_LEAGUES_LABELS.items()),
                               format_func=lambda x: x[1])
        if st.button("📡 Charger", disabled=not api_ok):
            with st.spinner("Chargement…"):
                data = api_get(f"/api/leagues/{league[0]}")
                if data:
                    for m in data.get("matches", []):
                        teams = m.get("teams", [])
                        home = next((t for t in teams if t.get("home")), {})
                        away = next((t for t in teams if not t.get("home")), {})
                        st.markdown(
                            f"**{home.get('name', '?')} {home.get('score', '')} — "
                            f"{away.get('score', '')} {away.get('name', '?')}**  "
                            f"*{m.get('date', '')[:10]}* — {m.get('status', '')}"
                        )

    st.markdown("---")
    st.subheader("📂 Matchs disponibles")
    if api_ok:
        data = api_get("/api/matches")
        if data:
            matches = data.get("matches", [])
            if matches:
                for m in matches:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"🎬 **{m['match_name']}** — `{m['match_id']}` ({m.get('size_mb', '?')} MB)")
                    with col2:
                        if st.button("Sélectionner", key=f"sel_{m['match_id']}"):
                            st.session_state.match_id = m["match_id"]
                            st.session_state.match_name = m["match_name"]
                            st.success(f"Match sélectionné : {m['match_name']}")
            else:
                st.info("Aucun match importé. Utilisez les onglets ci-dessus.")


# Constante pour les libellés de ligues (utilisée dans le tab_leagues)
RugbyDataConnector_LEAGUES_LABELS = {
    "top14": "Top 14 (France)",
    "premiership": "Premiership (Angleterre)",
    "super_rugby": "Super Rugby Pacific",
    "six_nations": "Six Nations",
    "world_cup": "Coupe du Monde",
    "champions_cup": "Champions Cup",
}


# ===========================================================================
# PAGE : Analyse temps réel
# ===========================================================================

elif page == "▶️ Analyse temps réel":
    st.header("▶️ Analyse en temps réel")

    if not st.session_state.match_id:
        st.warning("Veuillez d'abord sélectionner un match dans 📥 Sources")
        st.stop()

    st.markdown(f"**Match :** {st.session_state.match_name} (`{st.session_state.match_id}`)")

    # Métriques en temps réel
    col_phase, col_home, col_away, col_fps = st.columns(4)
    phase_placeholder = col_phase.empty()
    home_placeholder  = col_home.empty()
    away_placeholder  = col_away.empty()
    fps_placeholder   = col_fps.empty()

    # Insights
    st.markdown("### 💡 Key Insights")
    insights_placeholder = st.empty()

    # Derniers événements
    st.markdown("### ⚡ Événements récents")
    events_placeholder = st.empty()

    col_start, col_stop = st.columns(2)

    def render_metrics():
        phase_placeholder.metric("Phase de jeu", st.session_state.current_phase)
        home_placeholder.metric("Joueurs domicile", st.session_state.n_home)
        away_placeholder.metric("Joueurs visiteur", st.session_state.n_away)
        fps_placeholder.metric("FPS traitement", f"{st.session_state.processing_fps:.1f}")

    def render_insights():
        cards = "\n".join(
            f'<div class="insight-card">{i}</div>'
            for i in st.session_state.insights
        ) or "<i>Aucun insight pour le moment</i>"
        insights_placeholder.markdown(cards, unsafe_allow_html=True)

    def render_events():
        evs = st.session_state.events[-8:]
        if evs:
            html = "".join(
                f'<div class="event-card"><b>{e.get("event_type","?").upper()}</b> '
                f'@ {e.get("timestamp_s", 0):.1f}s — {e.get("description","")}</div>'
                for e in reversed(evs)
            )
            events_placeholder.markdown(html, unsafe_allow_html=True)
        else:
            events_placeholder.info("Aucun événement détecté")

    render_metrics()
    render_insights()
    render_events()

    if col_start.button("▶️ Démarrer l'analyse", disabled=st.session_state.is_analyzing or not api_ok):
        st.session_state.is_analyzing = True
        st.session_state.events.clear()
        st.session_state.patterns.clear()

        async def run_ws():
            uri = f"{WS_BASE}/ws/analysis/{st.session_state.match_id}"
            try:
                async with websockets.connect(uri, ping_interval=20) as ws:
                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                            data = json.loads(msg)
                            if data.get("status") == "completed":
                                break
                            # Mise à jour de l'état
                            st.session_state.current_phase = data.get("phase", "—")
                            st.session_state.phase_conf    = data.get("phase_confidence", 0)
                            st.session_state.n_home        = data.get("n_players_home", 0)
                            st.session_state.n_away        = data.get("n_players_away", 0)
                            st.session_state.processing_fps = data.get("processing_fps", 0)
                            st.session_state.insights      = data.get("key_insights", [])
                            st.session_state.zone_stats    = data.get("zone_stats", {})
                            evts = data.get("recent_events", [])
                            for ev in evts:
                                if ev not in st.session_state.events:
                                    st.session_state.events.append(ev)
                            pats = data.get("recent_patterns", [])
                            for p in pats:
                                if p not in st.session_state.patterns:
                                    st.session_state.patterns.append(p)
                            render_metrics()
                            render_insights()
                            render_events()
                        except asyncio.TimeoutError:
                            continue
            except Exception as e:
                st.error(f"Erreur WebSocket : {e}")
            finally:
                st.session_state.is_analyzing = False

        asyncio.run(run_ws())

    if col_stop.button("⏹️ Arrêter", disabled=not st.session_state.is_analyzing):
        st.session_state.is_analyzing = False


# ===========================================================================
# PAGE : Heatmaps
# ===========================================================================

elif page == "🗺️ Heatmaps":
    st.header("🗺️ Heatmaps — Utilisation de l'espace")

    if not st.session_state.match_id:
        st.warning("Veuillez sélectionner un match")
        st.stop()

    mode = st.radio(
        "Affichage",
        ["global", "home", "away", "ball"],
        format_func=lambda x: {
            "global": "Tous les joueurs",
            "home": "Équipe domicile",
            "away": "Équipe visiteur",
            "ball": "Trajectoire du ballon",
        }[x],
        horizontal=True,
    )

    if st.button("🔄 Actualiser la heatmap", disabled=not api_ok):
        with st.spinner("Génération…"):
            resp = requests.get(
                f"{API_BASE}/api/matches/{st.session_state.match_id}/heatmap",
                params={"mode": mode},
                timeout=30,
            )
            if resp.status_code == 200:
                import io
                from PIL import Image
                img = Image.open(io.BytesIO(resp.content))
                st.image(img, use_column_width=True)
            else:
                st.error(f"Erreur : {resp.text}")

    # Statistiques zones
    if st.session_state.zone_stats:
        st.markdown("### 📊 Occupation par zone")
        import pandas as pd
        rows = []
        for zone, stats in st.session_state.zone_stats.items():
            rows.append({
                "Zone": zone,
                "Domicile (%)": stats.get("home_pct", 0),
                "Visiteur (%)": stats.get("away_pct", 0),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Zone"))


# ===========================================================================
# PAGE : Événements
# ===========================================================================

elif page == "⚡ Événements":
    st.header("⚡ Timeline des événements")

    if not st.session_state.match_id or not api_ok:
        st.warning("Sélectionnez un match et assurez-vous que l'API est active")
        st.stop()

    data = api_get(f"/api/matches/{st.session_state.match_id}/events")
    events = data.get("events", []) if data else []

    if not events:
        st.info("Aucun événement détecté. Lancez l'analyse d'abord.")
    else:
        # Filtres
        types = list({e["event_type"] for e in events})
        selected = st.multiselect("Filtrer par type", types, default=types)
        filtered = [e for e in events if e["event_type"] in selected]

        st.markdown(f"**{len(filtered)} événement(s)**")

        # Timeline
        import pandas as pd
        df = pd.DataFrame(filtered)
        if not df.empty:
            df["time"] = df["timestamp_s"].apply(
                lambda s: f"{int(s//60):02d}:{int(s%60):02d}"
            )
            for _, row in df.sort_values("timestamp_s").iterrows():
                severity_color = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
                icon = severity_color.get(row.get("severity", "info"), "🔵")
                st.markdown(
                    f"{icon} **{row['time']}** — "
                    f"**{row['event_type'].upper()}** "
                    f"{'— Équipe: ' + row['team'] if row.get('team') else ''} "
                    f"— {row.get('description', '')}"
                )


# ===========================================================================
# PAGE : Patterns
# ===========================================================================

elif page == "🧩 Patterns":
    st.header("🧩 Patterns tactiques détectés")

    if not st.session_state.match_id or not api_ok:
        st.warning("Sélectionnez un match et assurez-vous que l'API est active")
        st.stop()

    data = api_get(f"/api/matches/{st.session_state.match_id}/patterns")
    patterns = data.get("patterns", []) if data else []

    if not patterns:
        st.info("Aucun pattern détecté. Lancez l'analyse d'abord.")
    else:
        import pandas as pd
        df = pd.DataFrame(patterns)
        df["time"] = df["timestamp_s"].apply(
            lambda s: f"{int(s//60):02d}:{int(s%60):02d}"
        )

        # Comptage par pattern
        counts = df["pattern"].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Fréquence des patterns")
            st.bar_chart(counts)
        with col2:
            st.markdown("### Détail")
            for _, row in df.sort_values("timestamp_s").iterrows():
                st.markdown(
                    f"🕐 **{row['time']}** — `{row['pattern']}` "
                    f"(conf: {row['confidence']:.0%}) — {row['description']}"
                )


# ===========================================================================
# PAGE : Insights
# ===========================================================================

elif page == "📈 Insights":
    st.header("📈 Key Insights automatiques")

    if not st.session_state.match_id or not api_ok:
        st.warning("Sélectionnez un match et assurez-vous que l'API est active")
        st.stop()

    data = api_get(f"/api/matches/{st.session_state.match_id}/insights")
    if data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Phase actuelle", data.get("current_phase", "—"))
        col2.metric("Événements détectés", data.get("n_events", 0))
        col3.metric("Patterns détectés", data.get("n_patterns", 0))

        st.markdown("### 💡 Insights")
        for insight in data.get("insights", []):
            st.markdown(
                f'<div class="insight-card">{insight}</div>',
                unsafe_allow_html=True,
            )

        zone_stats = data.get("zone_stats", {})
        if zone_stats:
            st.markdown("### 🗺️ Occupation des zones")
            import pandas as pd
            df = pd.DataFrame([
                {"Zone": z, "Domicile (%)": v["home_pct"], "Visiteur (%)": v["away_pct"]}
                for z, v in zone_stats.items()
            ])
            st.dataframe(df, use_container_width=True)


# ===========================================================================
# PAGE : Statistiques
# ===========================================================================

elif page == "📋 Statistiques":
    st.header("📋 Tableau de bord — Statistiques match")

    if not st.session_state.match_id or not api_ok:
        st.warning("Sélectionnez un match et assurez-vous que l'API est active")
        st.stop()

    events_data  = api_get(f"/api/matches/{st.session_state.match_id}/events")
    patterns_data = api_get(f"/api/matches/{st.session_state.match_id}/patterns")
    events   = events_data.get("events", [])   if events_data   else []
    patterns = patterns_data.get("patterns", []) if patterns_data else []

    # Compteurs
    tries = [e for e in events if e["event_type"] == "try"]
    scrums = [e for e in events if e["event_type"] == "scrum"]
    lineouts = [e for e in events if e["event_type"] == "lineout"]
    kicks = [e for e in events if e["event_type"] == "kick"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Essais", len(tries))
    c2.metric("⚙️ Mêlées", len(scrums))
    c3.metric("📍 Touches", len(lineouts))
    c4.metric("👟 Coups de pied", len(kicks))

    if events:
        import pandas as pd
        df_ev = pd.DataFrame(events)

        st.markdown("### Distribution des phases de jeu")
        phase_counts = df_ev["event_type"].value_counts()
        st.bar_chart(phase_counts)

        if st.session_state.zone_stats:
            st.markdown("### Occupation spatiale")
            df_zones = pd.DataFrame([
                {"Zone": z, "Domicile": v["home_pct"], "Visiteur": v["away_pct"]}
                for z, v in st.session_state.zone_stats.items()
            ])
            st.dataframe(df_zones.set_index("Zone"), use_container_width=True)

    if patterns:
        import pandas as pd
        st.markdown("### Patterns tactiques")
        df_pat = pd.DataFrame(patterns)
        st.dataframe(df_pat[["pattern", "confidence", "timestamp_s", "description"]],
                     use_container_width=True)
