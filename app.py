import logging

import joblib
import pandas as pd
import streamlit as st

from dap.analysis.regression import (
    DISEASE_COL_MAP,
    get_most_common_disease_for_breed,
    run_regression,
)
from dap.config import DATA_PATH, MODEL_PATH
from dap.data.loader import load_data
from dap.models.cslb_model import predict_cslb, train_cslb_model
from dap.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DAP – Analytic Avengers",
    page_icon="./asset/dap.png",
    layout="wide",
)
st.markdown("""
        <style>
        /* =========================================
           DYNAMIC THEME VARIABLES
           ========================================= */
        :root {
            --text-main: #0f172a;       /* Dark slate for high contrast */
            --text-sub: #64748b;        /* Medium slate for reading */
            --card-bg: #ffffff;         /* Clean white */
            --card-border: #e2e8f0;     /* Light gray border */
            --shadow-color: rgba(15, 23, 42, 0.08);
            --btn-bg: #ffffff;
            --btn-border: #cbd5e1;
            --accent-gradient: linear-gradient(135deg, #6366f1, #0ea5e9); /* Indigo to Sky */
            --btn-hover-bg: #f8fafc;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --text-main: #f8fafc;       /* Off-white for dark mode */
                --text-sub: #94a3b8;        /* Light slate */
                --card-bg: #1e293b;         /* Deep slate background */
                --card-border: #334155;     /* Dark border */
                --shadow-color: rgba(0, 0, 0, 0.4);
                --btn-bg: #0f172a;
                --btn-border: #334155;
                --accent-gradient: linear-gradient(135deg, #818cf8, #38bdf8); /* Brighter Indigo to Sky */
                --btn-hover-bg: #1e293b;
            }
        }

        /* =========================================
           COMPONENT STYLING
           ========================================= */
        /* Gradient Title */
        .main-header {
            font-size: 3.4rem;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            padding-top: 1rem;
        }

        /* Refined Subtext */
        .home-subtext {
            font-size: 1.1rem;
            color: var(--text-sub);
            margin-bottom: 2.5rem;
            font-weight: 400;
            letter-spacing: 0.3px;
        }

        /* Modern Information Card */
        .guide-card {
            background-color: #f5f9ff;
            padding: 1.8rem;
            border-radius: 20px;
            border: 1px solid #4a90e2;
            box-shadow: 0 10px 30px var(--shadow-color);
            color: #1f2937;
            position: relative;
            overflow: hidden;
        }

        .guide-card h4 {
            color: #1f2937 !important;
        }

        .guide-card p {
            color: #374151 !important;
        }

        .guide-card b {
            color: #111827 !important;
        }
        
        /* Premium Button Overrides */
        div.stButton > button {
            border-radius: 16px;
            padding: 1.2rem 1rem;
            min-height: 110px;
            background-color: var(--btn-bg);
            color: var(--text-main);
            border: 1px solid var(--btn-border);
            box-shadow: 0 4px 14px var(--shadow-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Button text font size */
        div.stButton > button p {
            font-size: 18px !important;
            font-weight: 600;
        }
    
        /* Caption text font size */
        [data-testid="stCaptionContainer"] p {
            font-size: 20px !important;
        }

        /* Button Hover State */
        div.stButton > button:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px var(--shadow-color);
            border-color: #6366f1; /* Accent color border on hover */
            background-color: var(--btn-hover-bg);
        }

        /* Button Active/Click State */
        div.stButton > button:active {
            transform: translateY(0px);
            box-shadow: 0 4px 10px var(--shadow-color);
        }
        </style>
        """, unsafe_allow_html=True)

col_left, col_right = st.columns([4, 1])
with col_left:
    st.markdown('<div class="main-header">DAP by Analytic Avengers 🐾</div>', unsafe_allow_html=True)
    st.caption("Predictive health analysis powered by the Dog Aging Project dataset. A Journey in Canine Wellness!")
with col_right:
    st.image("./asset/dap.png",width=200)

@st.cache_data(show_spinner="Loading the Dog Aging Project dataset...")
def get_app_data():
    return load_data(DATA_PATH)

@st.cache_resource(show_spinner="Loading trained CSLB model...")
def get_cslb_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}\n\n"
            "Train the model first by running:\n"
            "python scripts/train_cslb.py"
        )

    return joblib.load(MODEL_PATH)

def init_state() -> None:
    defaults = {
        "page": "home",
        "cslb_predicted": None,
        "cslb_inputs": None,
        "fu_submitted": False,
        "fu_pace": 0,
        "fu_stare": 0,
        "fu_defecate": 0,
        "fu_food": 0,
        "fu_recognize": 0,
        "fu_active": 0,
        "reg_results": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_home() -> None:
    st.session_state.page = "home"
    st.session_state.cslb_predicted = None
    st.session_state.cslb_inputs = None
    st.session_state.fu_submitted = False
    st.session_state.reg_results = None


def go_cslb() -> None:
    st.session_state.page = "cslb"
    st.session_state.cslb_predicted = None
    st.session_state.cslb_inputs = None
    st.session_state.fu_submitted = False


def go_regression() -> None:
    st.session_state.page = "regression"
    st.session_state.reg_results = None


def render_header() -> None:
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.4rem;
                font-weight: 800;
                color: #2563eb;
                margin-bottom: 0.2rem;
            }

            .subtitle {
                font-size: 1.05rem;
                color: #64748b;
                margin-bottom: 1.2rem;
            }

            .disclaimer {
                padding: 0.8rem;
                border-radius: 0.5rem;
                background-color: #fef3c7;
                color: #78350f;
                border: 1px solid #f59e0b;
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    # Layout setup
    col_form, gap, col_info = st.columns([3.5, 1, 2])

    with col_form:
        st.markdown("#### 🔍 Choose an analysis")
        st.write("")  # Small spacer

        task_col1, task_col2 = st.columns(2)

        with task_col1:
            if st.button(
                    "🧠 Cognitive Dysfunction\nPrediction",
                    use_container_width=True
            ):
                go_cslb()
                st.rerun()

        with task_col2:
            if st.button(
                    "🔬 Disease Regression\nAnalysis",
                    use_container_width=True
            ):
                go_regression()
                st.rerun()

    with col_info:
        st.markdown("""
                        <div class="guide-card">
                            <h4>🐾 User Guide</h4>
                            <p>- Select the analysis you want to run.</p>
                            <p><b>Cognitive dysfunction prediction:</b><br>
                            - Complete all questionnaires about your dog. Depending on the dogs condition, you may need observe for 6 months and then answer 6 more followup questions.</p>
                            <p><b>Disease regression analysis:</b><br>
                            - Choose your dog breed. <br>
                            - Select a disease and choose either one or more variables.</p>
                        </div>
                        """, unsafe_allow_html=True)


def render_cslb_page() -> None:
    if st.button("← Back to Home"):
        go_home()
        st.rerun()

    st.header("🧠 Cognitive Dysfunction Prediction")
    st.caption(
        "Estimate a Cognitive Dysfunction Syndrome score using CSLB-related "
        "questionnaire features."
    )

    st.markdown("---")

    if st.session_state.cslb_predicted is None:
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("🐾 Basic information")

            dog_age = st.number_input(
                "Dog age (years)",
                min_value=0.0,
                max_value=18.0,
                value=5.0,
                step=0.5,
            )

            dog_weight = st.number_input(
                "Dog weight (lbs)",
                min_value=1.0,
                max_value=300.0,
                value=30.0,
                step=0.5,
            )

            st.subheader("🏃 Activity and health")

            activity_options = {
                "1 – Inactive / sedentary": 1,
                "2 – Somewhat active": 2,
                "3 – Moderately active": 3,
                "4 – Very active": 4,
                "5 – Extremely active": 5,
            }

            pa_activity = activity_options[
                st.selectbox(
                    "Activity level over the past year",
                    list(activity_options.keys()),
                )
            ]

            eye_condition = st.selectbox(
                "Eye health condition?",
                ["No", "Yes"],
            )

            ear_condition = st.selectbox(
                "Ear health condition?",
                ["No", "Yes"],
            )

            eye_value = 0 if eye_condition == "No" else 2
            ear_value = 0 if ear_condition == "No" else 2

        with right_col:
            st.subheader("🧠 Behavioural questions (CSLB)")

            response_options = {
                "Never (0)": 0,
                "Rarely (1)": 1,
                "Sometimes (2)": 2,
                "Often (3)": 3,
                "Always (4)": 4,
            }

            option_labels = list(response_options.keys())

            cslb_pace = response_options[
                st.selectbox(
                    "Paces back and forth without purpose?",
                    option_labels,
                    key="cslb_pace",
                )
            ]

            cslb_stare = response_options[
                st.selectbox(
                    "Stares blankly into space or at walls?",
                    option_labels,
                    key="cslb_stare",
                )
            ]

            cslb_stuck = response_options[
                st.selectbox(
                    "Gets stuck behind furniture or in corners?",
                    option_labels,
                    key="cslb_stuck",
                )
            ]

            cslb_recognize = response_options[
                st.selectbox(
                    "Fails to recognise familiar people?",
                    option_labels,
                    key="cslb_recognize",
                )
            ]

            cslb_walk_walls = response_options[
                st.selectbox(
                    "Walks into walls or doors?",
                    option_labels,
                    key="cslb_walk_walls",
                )
            ]

            cslb_avoid = response_options[
                st.selectbox(
                    "Avoids being petted or has lost interest?",
                    option_labels,
                    key="cslb_avoid",
                )
            ]

            cslb_find_food = response_options[
                st.selectbox(
                    "Has difficulty finding food dropped on the floor?",
                    option_labels,
                    key="cslb_find_food",
                )
            ]

        st.markdown("---")

        if st.button("🔍 Predict Cognitive Health", type="primary"):
            input_data = pd.DataFrame(
                [
                    {
                        "dd_age_years": dog_age,
                        "dd_weight_lbs": dog_weight,
                        "pa_activity_level": pa_activity,
                        "hs_health_conditions_eye": eye_value,
                        "hs_health_conditions_ear": ear_value,
                        "cslb_pace": cslb_pace,
                        "cslb_stare": cslb_stare,
                        "cslb_stuck": cslb_stuck,
                        "cslb_recognize": cslb_recognize,
                        "cslb_walk_walls": cslb_walk_walls,
                        "cslb_avoid": cslb_avoid,
                        "cslb_find_food": cslb_find_food,
                    }
                ]
            )

            try:
                trained_model = get_cslb_model()

                prediction = predict_cslb(
                    trained_model=trained_model,
                    input_data=input_data,
                )

                st.session_state.cslb_predicted = float(prediction[0])

                st.session_state.cslb_inputs = {
                    "cslb_pace": cslb_pace,
                    "cslb_stare": cslb_stare,
                    "cslb_stuck": cslb_stuck,
                    "cslb_recognize": cslb_recognize,
                    "cslb_walk_walls": cslb_walk_walls,
                    "cslb_avoid": cslb_avoid,
                    "cslb_find_food": cslb_find_food,
                }

                logger.info(
                    "CSLB prediction generated: %.2f",
                    st.session_state.cslb_predicted,
                )

                st.rerun()

            except Exception as error:
                logger.exception("CSLB prediction failed")
                st.error(f"Prediction failed: {error}")

    else:
        render_cslb_result()


def render_cslb_result() -> None:
    predicted_score = st.session_state.cslb_predicted
    input_answers = st.session_state.cslb_inputs

    st.subheader("📊 Prediction result")
    st.metric("Predicted CSLB score", f"{predicted_score:.2f}")

    if predicted_score < 40:
        st.success(
            f"No strong indication of cognitive "
            "dysfunction from this model. Your dog is safe!"
        )

        if st.button("🔄 Run another assessment"):
            go_cslb()
            st.rerun()

    elif 40 <= predicted_score <= 60:
        st.warning(
            f"There is a possibility of your dog having symptoms of cognitive dysfunction! "
            f"Kindly monitor your dogs actions carefully for 6 months and answer some follow up questions:"
        )

        st.markdown("---")
        st.subheader("🗓️ Six-month follow-up questions")

        response_options = {
            "Never (0)": 0,
            "Rarely (1)": 1,
            "Sometimes (2)": 2,
            "Often (3)": 3,
            "Always (4)": 4,
        }

        option_labels = list(response_options.keys())

        with st.form("followup_form"):
            left_col, right_col = st.columns(2)

            with left_col:
                followup_pace = st.selectbox(
                    "(6 months) Paces back and forth?",
                    option_labels,
                    key="fu_pace_select",
                )

                followup_stare = st.selectbox(
                    "(6 months) Stares blankly?",
                    option_labels,
                    key="fu_stare_select",
                )

                followup_defecate = st.selectbox(
                    "(6 months) Defecates or urinates in an inappropriate place?",
                    option_labels,
                    key="fu_defecate_select",
                )

            with right_col:
                followup_food = st.selectbox(
                    "(6 months) Decreased interest in food?",
                    option_labels,
                    key="fu_food_select",
                )

                followup_recognize = st.selectbox(
                    "(6 months) Fails to recognise familiar people?",
                    option_labels,
                    key="fu_recognize_select",
                )

                followup_active = st.selectbox(
                    "(6 months) Less active or playful?",
                    option_labels,
                    key="fu_active_select",
                )

            followup_submitted = st.form_submit_button(
                "Calculate follow-up score",
                type="primary",
            )

        if followup_submitted:
            st.session_state.fu_pace = response_options[followup_pace]
            st.session_state.fu_stare = response_options[followup_stare]
            st.session_state.fu_defecate = response_options[followup_defecate]
            st.session_state.fu_food = response_options[followup_food]
            st.session_state.fu_recognize = response_options[
                followup_recognize
            ]
            st.session_state.fu_active = response_options[followup_active]
            st.session_state.fu_submitted = True
            st.rerun()

        if st.session_state.fu_submitted:
            final_score = (
                input_answers["cslb_pace"]
                + input_answers["cslb_stare"]
                + input_answers["cslb_stuck"]
                + input_answers["cslb_recognize"]
                + input_answers["cslb_walk_walls"]
                + input_answers["cslb_avoid"]
                + input_answers["cslb_find_food"]
                + st.session_state.fu_pace
                + st.session_state.fu_stare
                + st.session_state.fu_defecate
                + (2 * st.session_state.fu_food)
                + (3 * st.session_state.fu_recognize)
                + st.session_state.fu_active
            )

            st.markdown("---")
            st.subheader("📋 Follow-up CSLB score")
            st.metric("Follow-up score", final_score)

            if final_score < 50:
                st.success(
                    "NEW UPDATE :The follow-up CSLB score is below the educational threshold used in this application "
                    "\nNo need to worry! your dog has no signs of cognitive dysfunction!"
                )
            else:
                st.error(
                    "NEW UPDATE : The follow-up CSLB score is above the educational threshold "
                    "used in this application. He/she has symptoms of cognitive dysfunction! Kindly contact the vet as soon as possible"
                )

            if st.button("🔄 Start over"):
                go_home()
                st.rerun()

    else:
        st.error(
            f"Score {predicted_score:.1f}: this model indicates strong "
            "cognitive dysfunction-related patterns. Kindly contact the vet as soon as possible."
        )

        if st.button("🔄 Run another assessment"):
            go_cslb()
            st.rerun()


def render_regression_page(
    data: pd.DataFrame,
    breed_list: list[str],
) -> None:
    if st.button("← Back to Home"):
        go_home()
        st.rerun()

    st.header("🔬 Disease Regression Analysis")

    st.caption(
        "Select a breed, a disease category, and one or more variable groups "
        "to examine statistically significant associations."
    )

    breed = st.selectbox(
        "🐕 Select a dog's breed",
        options=["— select a breed —"] + breed_list,
        index=0,
    )

    breed_summary = get_most_common_disease_for_breed(data, breed)
    st.info(breed_summary)

    st.markdown("---")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        disease_choice = st.selectbox(
            "Select a disease",
            options=list(DISEASE_COL_MAP.keys()),
            format_func=lambda item: item.replace("_", " ").capitalize(),
        )

        variable_groups = st.multiselect(
            "Select variable group(s)",
            options=[
                "diet",
                "physical_activity",
                "behavior",
                "environment",
            ],
            default=["diet"],
            format_func=lambda item: item.replace("_", " ").capitalize(),
        )

        run_analysis = st.button(
            "🔍 Run Regression Analysis",
            type="primary",
        )

    with right_col:
        if run_analysis:
            if not variable_groups:
                st.warning("Select at least one variable group.")
            else:
                disease_column = DISEASE_COL_MAP[disease_choice]

                if disease_column not in data.columns:
                    st.error(
                        f"Required disease column is not in the dataset: "
                        f"`{disease_column}`"
                    )
                else:
                    all_results = {}

                    for variable_group in variable_groups:
                        try:
                            with st.spinner(
                                "Running analysis for "
                                f"{variable_group.replace('_', ' ')}..."
                            ):
                                suggestions, findings = run_regression(
                                    disease_key=disease_choice,
                                    variable_group=variable_group,
                                    data=data,
                                )

                            all_results[variable_group] = {
                                "status": "ok",
                                "suggestions": suggestions,
                                "findings": findings,
                            }

                        except Exception as error:
                            logger.exception(
                                "Regression failed for %s",
                                variable_group,
                            )

                            all_results[variable_group] = {
                                "status": "error",
                                "error": str(error),
                            }

                    st.session_state.reg_results = {
                        "disease": disease_choice,
                        "groups": all_results,
                    }

        if st.session_state.reg_results:
            results = st.session_state.reg_results

            st.subheader(
                f"Results for: {results['disease'].replace('_', ' ').capitalize()}"
            )

            for group_name, result in results["groups"].items():
                st.markdown(
                    f"### 📌 {group_name.replace('_', ' ').capitalize()}"
                )

                if result["status"] == "error":
                    st.error(
                        f"Regression failed for {group_name}: "
                        f"{result['error']}"
                    )
                    continue

                findings = result["findings"]
                suggestions = result["suggestions"]

                if not findings:
                    st.info(
                        "No statistically significant associations were found "
                        "under the current analysis settings."
                    )
                else:
                    findings_tab, suggestions_tab = st.tabs(
                        [
                            "📊 Statistical Findings",
                            "💡 Interpretation Notes",
                        ]
                    )

                    with findings_tab:
                        for finding in findings:
                            st.markdown(f"- {finding}")

                    with suggestions_tab:

                        for suggestion in suggestions:
                            st.markdown(f"- {suggestion}")

                st.divider()


def main() -> None:
    init_state()
    render_header()

    try:
        data, breed_list = get_app_data()

    except (FileNotFoundError, ValueError) as error:
        logger.exception("Dataset loading failed")

        st.error("Dataset setup is incomplete.")

        st.code(
            "1. Obtain approved Dog Aging Project data access.\n"
            "2. Run sql/create_final_dataset.sql in your SQL environment.\n"
            "3. Export the result as data/final.csv.\n"
            "4. Run: streamlit run app.py"
        )

        st.info(str(error))
        st.stop()

    if st.session_state.page == "home":
        render_home()

    elif st.session_state.page == "cslb":
        render_cslb_page()

    elif st.session_state.page == "regression":
        render_regression_page(data, breed_list)

    st.markdown("---")
    st.markdown(
        """
        <div class="disclaimer">
        <b>Educational disclaimer:</b> This application is an academic
        data-science project. It is not a veterinary diagnostic system and
        must not replace professional veterinary advice, diagnosis, or treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "© Rafid Anwar · MSc ICE · Educational project using "
        "Dog Aging Project data - [Website](https://dogagingproject.org)"
    )

if __name__ == "__main__":
    main()