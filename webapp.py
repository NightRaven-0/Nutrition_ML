# webapp.py
from pathlib import Path

import joblib
import pandas as pd
from nicegui import ui

from src.rule_engine import check_anemia
from src.recommend import acute_text, build_recommendation

BASE = Path(__file__).resolve().parent
ACUTE_PATH = BASE / "models" / "acute_model.pkl"
STUNT_PATH = BASE / "models" / "stunting_model.pkl"
SCALER_PATH = BASE / "data" / "processed" / "scaler.pkl"

# If you put Geist fonts here, they’ll be served at /fonts/GeistVF.woff2 etc:
FONTS_DIR = BASE / "fonts"


THEME_CSS = r"""
<style>
/* Geist (optional): place files in ./fonts */
@font-face {
  font-family: 'Geist';
  src: url('/fonts/GeistVF.woff2') format('woff2');
  font-weight: 100 900;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: 'Geist Mono';
  src: url('/fonts/GeistMonoVF.woff2') format('woff2');
  font-weight: 100 900;
  font-style: normal;
  font-display: swap;
}

/* Your token scheme (Tailwind-style HSL variables) */
:root{
  --background: 0 0% 100%;
  --foreground: 0 0% 3.9%;
  --primary: 0 0% 9%;
  --primary-foreground: 0 0% 98%;
  --secondary: 0 0% 80.1%;
  --secondary-foreground: 0 0% 9%;
  --muted: 0 0% 80.1%;
  --muted-foreground: 0 0% 45.1%;
  --accent: 0 0% 80.1%;
  --accent-foreground: 0 0% 9%;
  --additive: 112 50% 36%;
  --additive-foreground: 0 0% 9%;
  --destructive: 0 84.2% 60.2%;
  --destructive-foreground: 0 0% 98%;
  --border: 0 0% 89.8%;
  --ring: 0 0% 3.9%;

  /* map into Quasar theme vars used by NiceGUI */
  --q-primary: hsl(var(--primary));
  --q-secondary: hsl(var(--secondary));
  --q-accent: hsl(var(--accent));
  --q-positive: hsl(var(--additive));
  --q-negative: hsl(var(--destructive));
  --q-info: hsl(var(--muted-foreground));
  --q-warning: hsl(var(--accent));
}

html.dark, body.body--dark{
  --background: 240 22.7% 8.6%;
  --foreground: 160 100% 45%;
  --primary: 0 0% 98%;
  --primary-foreground: 0 0% 9%;
  --secondary: 0 0% 14.9%;
  --secondary-foreground: 160 100% 45%;
  --muted: 0 0% 14.9%;
  --muted-foreground: 0 0% 63.9%;
  --accent: 0 0% 14.9%;
  --accent-foreground: 0 0% 98%;
  --additive: 112 50% 36%;
  --additive-foreground: 0 0% 9%;
  --destructive: 0 62.8% 30.6%;
  --destructive-foreground: 0 0% 98%;
  --border: 0 0% 14.9%;
  --ring: 0 0% 83.1%;

  --q-primary: hsl(var(--primary));
  --q-secondary: hsl(var(--secondary));
  --q-accent: hsl(var(--accent));
  --q-positive: hsl(var(--additive));
  --q-negative: hsl(var(--destructive));
  --q-info: hsl(var(--muted-foreground));
  --q-warning: hsl(var(--accent));
}

html, body{
  background: hsl(var(--background));
  color: hsl(var(--foreground));
  font-family: Geist, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif;
}

.q-header{
  background: hsl(var(--background)) !important;
  color: hsl(var(--foreground)) !important;
  border-bottom: 1px solid hsl(var(--border)) !important;
}

.q-card{
  background: hsl(var(--background)) !important;
  color: hsl(var(--foreground)) !important;
  border: 1px solid hsl(var(--border)) !important;
  border-radius: 16px !important;
}

.q-field--outlined .q-field__control{
  border-color: hsl(var(--border)) !important;
}

.q-btn--unelevated{
  border-radius: 12px !important;
}

.q-chip{
  border-radius: 999px !important;
  border: 1px solid hsl(var(--border)) !important;
}
</style>
"""


def load_artifacts():
    missing = [p for p in (ACUTE_PATH, STUNT_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(str(p) for p in missing))
    return (
        joblib.load(str(ACUTE_PATH)),
        joblib.load(str(STUNT_PATH)),
        joblib.load(str(SCALER_PATH)),
    )


def assess(acute_model, stunting_model, scaler, age, sex, weight, height, muac, hb):
    bmi = weight / ((height / 100.0) ** 2)

    sample_df = pd.DataFrame(
        [[age, sex, weight, height, muac, hb, bmi]],
        columns=["age", "sex", "weight", "height", "muac", "hb", "bmi"],
    )

    sample_scaled = scaler.transform(sample_df)
    acute_pred = int(acute_model.predict(sample_scaled)[0])
    stunting_pred = bool(int(stunting_model.predict(sample_scaled)[0]))
    anemia_flag = bool(check_anemia(hb))
    recs = build_recommendation(acute_pred, stunting_pred, anemia_flag)

    return {
        "acute_pred": acute_pred,
        "acute_text": acute_text(acute_pred),
        "stunting_pred": stunting_pred,
        "anemia_flag": anemia_flag,
        "bmi": float(bmi),
        "recs": recs,
    }


ui.add_head_html(THEME_CSS)
ui.page_title("Nutrition ML — Child Assessment")

# Serve /fonts if you add Geist fonts locally
try:
    if FONTS_DIR.exists():
        from fastapi.staticfiles import StaticFiles
        ui.get_app().mount("/fonts", StaticFiles(directory=str(FONTS_DIR)), name="fonts")
except Exception:
    pass

dm = ui.dark_mode(value=False)


def set_dark(v: bool):
    dm.value = v
    ui.run_javascript(f"document.documentElement.classList.toggle('dark', {str(v).lower()});")


acute_model = stunting_model = scaler = None
load_error = None
try:
    acute_model, stunting_model, scaler = load_artifacts()
except Exception as e:
    load_error = str(e)

with ui.header().classes("items-center justify-between q-px-md"):
    with ui.row().classes("items-center gap-2"):
        ui.icon("health_and_safety").classes("text-h6")
        ui.label("Nutrition ML — Child Assessment").classes("text-h6")
    with ui.row().classes("items-center gap-2"):
        ui.switch("Dark", value=False, on_change=lambda e: set_dark(bool(e.value))).props("dense")
        ui.icon("palette").classes("opacity-70")

with ui.column().classes("w-full max-w-3xl mx-auto q-pa-md gap-4"):

    if load_error:
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("error").classes("text-negative text-h6")
                ui.label("Model files not found / failed to load").classes("text-h6")
            ui.separator()
            ui.markdown(f"```\n{load_error}\n```")
            ui.markdown(
                f"- `{ACUTE_PATH}`\n- `{STUNT_PATH}`\n- `{SCALER_PATH}`\n\n"
                "Generate/train these files, then refresh."
            )
    else:
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("person").classes("text-h6")
                ui.label("Child Inputs").classes("text-h6")
            ui.separator()

            with ui.row().classes("w-full gap-3 items-start"):
                age_in = ui.number("Age (months)", min=0, step=1, value=24).props("outlined dense").classes("w-56")
                with age_in.add_slot("prepend"):
                    ui.icon("calendar_month")

                sex_in = ui.radio(
                    {0: "Female", 1: "Male"},
                    value=0,
                ).props("inline dense color=primary").classes("q-mt-sm")

            with ui.row().classes("w-full gap-3"):
                wt_in = ui.number("Weight (kg)", min=0, step=0.1, value=12.0).props("outlined dense").classes("w-56")
                with wt_in.add_slot("prepend"):
                    ui.icon("monitor_weight")

                ht_in = ui.number("Height (cm)", min=0, step=0.1, value=85.0).props("outlined dense").classes("w-56")
                with ht_in.add_slot("prepend"):
                    ui.icon("straighten")

            with ui.row().classes("w-full gap-3"):
                muac_in = ui.number("MUAC (mm)", min=0, step=0.1, value=130.0).props("outlined dense").classes("w-56")
                with muac_in.add_slot("prepend"):
                    ui.icon("fitness_center")

                hb_in = ui.number("Hemoglobin", min=0, step=0.1, value=11.5).props("outlined dense").classes("w-56")
                with hb_in.add_slot("prepend"):
                    ui.icon("bloodtype")

            result_wrap = ui.column().classes("w-full gap-3 hidden")
            report_card = ui.card().classes("w-full")
            recs_card = ui.card().classes("w-full")

            def show_result(r):
                result_wrap.classes(remove="hidden")

                report_card.clear()
                recs_card.clear()

                with report_card:
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("assignment").classes("text-h6")
                        ui.label("Assessment Report").classes("text-h6")
                    ui.separator()

                    with ui.row().classes("w-full gap-2 q-mt-sm"):
                        ui.chip(f"Acute: {r['acute_text']}", icon="warning").props("outline color=warning")
                        ui.chip(f"Stunting: {'Yes' if r['stunting_pred'] else 'No'}", icon="height").props("outline color=info")
                        ui.chip(f"Anemia: {'Yes' if r['anemia_flag'] else 'No'}", icon="bloodtype").props("outline color=negative" if r["anemia_flag"] else "outline color=positive")
                        ui.chip(f"BMI: {r['bmi']:.2f}", icon="insights").props("outline color=accent")

                with recs_card:
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("restaurant_menu").classes("text-h6")
                        ui.label("Dietary Focus").classes("text-h6")
                    ui.separator()

                    if not r["recs"]:
                        ui.label("No recommendations returned.").classes("opacity-70 q-mt-sm")
                    else:
                        for t in r["recs"]:
                            with ui.card().classes("w-full q-mt-sm"):
                                with ui.row().classes("items-start gap-2"):
                                    ui.icon("check_circle").classes("q-mt-xs").style("color: hsl(var(--additive));")
                                    ui.label(t).classes("text-body1")

            def on_submit():
                try:
                    age = int(age_in.value)
                    sex = int(sex_in.value)
                    weight = float(wt_in.value)
                    height = float(ht_in.value)
                    muac = float(muac_in.value)
                    hb = float(hb_in.value)

                    if height <= 0 or weight <= 0:
                        ui.notify("Height and weight must be > 0", type="negative")
                        return

                    r = assess(acute_model, stunting_model, scaler, age, sex, weight, height, muac, hb)
                    show_result(r)
                    ui.notify("Assessment complete", type="positive")
                except Exception as e:
                    ui.notify(f"Error: {e}", type="negative")

            def on_reset():
                age_in.value = 24
                sex_in.value = 0
                wt_in.value = 12.0
                ht_in.value = 85.0
                muac_in.value = 130.0
                hb_in.value = 11.5
                result_wrap.classes(add="hidden")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Reset", on_click=on_reset, icon="restart_alt").props("outline color=secondary")
                ui.button("Assess", on_click=on_submit, icon="analytics").props("unelevated color=primary")

            with result_wrap:
                report_card
                recs_card

ui.run(host="0.0.0.0", port=8081)
