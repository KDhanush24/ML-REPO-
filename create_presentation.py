"""
Presentation Generator for B.Tech Machine Learning Project:
'Stock Return Prediction Using Ridge & Lasso Regression'

Generates an editable 21-slide 16:9 widescreen PowerPoint presentation
(.pptx) with modern academic styling, tables, architecture diagrams,
embedded evaluation charts, and comprehensive presenter speaker notes.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PPTX_PATH = PROJECT_ROOT / "Stock_Return_Prediction_Project_Presentation.pptx"
GRAPHS_DIR = PROJECT_ROOT / "outputs" / "graphs"

import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# ---------------------------------------------------------------------------
# Design System & Color Palette
# ---------------------------------------------------------------------------
NAVY = RGBColor(15, 32, 66)         # #0F2042 - Primary Dark & Header
ROYAL_BLUE = RGBColor(30, 64, 175)  # #1E40AF - Accent Primary
TEAL = RGBColor(13, 148, 136)       # #0D9488 - Accent Secondary / Success
SLATE_DARK = RGBColor(30, 41, 59)   # #1E293B - Charcoal Body Text
SLATE_MUTED = RGBColor(100, 116, 139) # #64748B - Subtitles & Footers
SLATE_BORDER = RGBColor(203, 213, 225) # #CBD5E1 - Card Borders
BG_LIGHT = RGBColor(248, 250, 252)  # #F8FAFC - Slide Background
CARD_WHITE = RGBColor(255, 255, 255)# #FFFFFF - Card Background
CARD_ACCENT = RGBColor(240, 249, 255) # #F0F9FF - Callout Card Fill
RED_ACCENT = RGBColor(220, 38, 38)  # #DC2626 - Ridge/Lasso Accent

FONT_TITLE = "Calibri"
FONT_BODY = "Calibri"

TOTAL_SLIDES = 21


def create_presentation() -> pptx.Presentation:
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def add_slide_background(slide, color=BG_LIGHT):
    """Fill slide with background color."""
    bg_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5)
    )
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = color
    bg_shape.line.fill.background()
    return bg_shape


def add_header(slide, title_text: str, category_text: str, subtitle_text: str = ""):
    """Standard header banner for content slides."""
    # Top subtle colored accent bar
    top_bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.08)
    )
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = ROYAL_BLUE
    top_bar.line.fill.background()

    # Header text box
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.25), Inches(11.733), Inches(1.15))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    # Category Pill / Tag
    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = FONT_TITLE
    p_cat.font.size = Pt(9.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = TEAL
    p_cat.space_after = Pt(2)

    # Title
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_TITLE
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY

    # Subtitle if present
    if subtitle_text:
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle_text
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = SLATE_MUTED


def add_footer(slide, current_slide: int):
    """Standard bottom footer for content slides."""
    # Footer line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(7.05), Inches(11.733), Inches(0.02)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = SLATE_BORDER
    line.line.fill.background()

    # Footer Text Box
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(7.12), Inches(11.733), Inches(0.3))
    tf = tb.text_frame
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_left = tf.paragraphs[0]
    p_left.text = "Stock Return Prediction Using Ridge & Lasso Regression  |  B.Tech ML Project"
    p_left.font.name = FONT_BODY
    p_left.font.size = Pt(9)
    p_left.font.color.rgb = SLATE_MUTED

    # Slide Number Box
    tb_num = slide.shapes.add_textbox(Inches(11.5), Inches(7.12), Inches(1.033), Inches(0.3))
    tf_num = tb_num.text_frame
    tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
    p_num = tf_num.paragraphs[0]
    p_num.alignment = PP_ALIGN.RIGHT
    p_num.text = f"{current_slide} / {TOTAL_SLIDES}"
    p_num.font.name = FONT_BODY
    p_num.font.size = Pt(9)
    p_num.font.bold = True
    p_num.font.color.rgb = NAVY


def add_card(slide, left: float, top: float, width: float, height: float,
             bg_color=CARD_WHITE, border_color=SLATE_BORDER, corner_radius=None):
    """Draw a clean card container shape."""
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if corner_radius else MSO_SHAPE.RECTANGLE
    card = slide.shapes.add_shape(
        shape_type, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
    else:
        card.line.fill.background()
    return card


def add_bullet_points(tf, points, pt_size=12, space_after=6, bold_first_n_words=0):
    """Add bullet points to a text frame with professional formatting."""
    for i, pt_text in enumerate(points):
        p = tf.add_paragraph() if i > 0 or len(tf.paragraphs[0].text) > 0 else tf.paragraphs[0]
        p.space_after = Pt(space_after)
        p.font.name = FONT_BODY
        p.font.size = Pt(pt_size)
        p.font.color.rgb = SLATE_DARK
        p.level = 0

        # Handle bold prefix if specified
        if bold_first_n_words > 0 and ":" in pt_text:
            prefix, rest = pt_text.split(":", 1)
            p.text = ""
            r1 = p.add_run()
            r1.text = "• " + prefix + ":"
            r1.font.bold = True
            r1.font.color.rgb = NAVY
            r2 = p.add_run()
            r2.text = rest
            r2.font.bold = False
            r2.font.color.rgb = SLATE_DARK
        else:
            p.text = "• " + pt_text


def add_speaker_notes(slide, notes_text: str):
    """Add conversational presenter speaker notes to the slide."""
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = notes_text.strip()


def add_table(slide, left: float, top: float, width: float, height: float,
              headers: list, rows: list, col_widths: list = None):
    """Create a styled academic table."""
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(
        num_rows, num_cols, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    table = table_shape.table

    if col_widths and len(col_widths) == num_cols:
        for idx, w in enumerate(col_widths):
            table.columns[idx].width = Inches(w)

    # Style Header Row
    for col_idx, header in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = str(header)
        p.font.name = FONT_TITLE
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    # Style Data Rows
    for row_idx, row_data in enumerate(rows):
        bg = RGBColor(255, 255, 255) if row_idx % 2 == 0 else RGBColor(241, 245, 249)
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = str(val)
            p.font.name = FONT_BODY
            p.font.size = Pt(10.5)
            p.font.color.rgb = SLATE_DARK
            # Left-align text column, center-align numbers
            if col_idx == 0:
                p.alignment = PP_ALIGN.LEFT
            else:
                p.alignment = PP_ALIGN.CENTER

    return table_shape


# ===========================================================================
# SLIDE BUILDERS (All 21 Slides in Exact Order)
# ===========================================================================

def build_slide_1_title(prs):
    """SLIDE 1: Title Slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    # Deep Navy Premium Background
    add_slide_background(slide, color=NAVY)

    # Decorative top and bottom accent lines
    top_acc = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.12))
    top_acc.fill.solid()
    top_acc.fill.fore_color.rgb = TEAL
    top_acc.line.fill.background()

    # Left content box - Title & Subtitle
    tb_title = slide.shapes.add_textbox(Inches(0.9), Inches(1.2), Inches(7.5), Inches(3.2))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = 0

    p_badge = tf_title.paragraphs[0]
    p_badge.text = "B.TECH MACHINE LEARNING MINI-PROJECT EVALUATION"
    p_badge.font.name = FONT_TITLE
    p_badge.font.size = Pt(11)
    p_badge.font.bold = True
    p_badge.font.color.rgb = TEAL
    p_badge.space_after = Pt(12)

    p_main = tf_title.add_paragraph()
    p_main.text = "Stock Return Prediction Using Ridge & Lasso Regression"
    p_main.font.name = FONT_TITLE
    p_main.font.size = Pt(30)
    p_main.font.bold = True
    p_main.font.color.rgb = RGBColor(255, 255, 255)
    p_main.space_after = Pt(14)

    p_sub = tf_title.add_paragraph()
    p_sub.text = "An End-to-End Educational Web Application for Automated Stock CSV Analysis, Feature Engineering, Regularization Modeling, and Next-Trading-Day Return Forecasting"
    p_sub.font.name = FONT_BODY
    p_sub.font.size = Pt(13)
    p_sub.font.color.rgb = RGBColor(203, 213, 225)

    # Right side card - Student & College Metadata Box
    card_info = add_card(slide, 8.8, 1.2, 3.65, 5.0, bg_color=RGBColor(26, 44, 76), border_color=ROYAL_BLUE)
    tb_info = slide.shapes.add_textbox(Inches(9.1), Inches(1.4), Inches(3.1), Inches(4.5))
    tf_info = tb_info.text_frame
    tf_info.word_wrap = True
    tf_info.margin_left = tf_info.margin_top = 0

    meta_items = [
        ("PROJECT METADATA", "Academic Presentation"),
        ("Student Name(s)", "[ENTER STUDENT NAME(S)]"),
        ("Roll Number(s)", "[ENTER ROLL NUMBER(S)]"),
        ("Project Guide", "[ENTER GUIDE NAME]"),
        ("Department", "[ENTER DEPARTMENT]"),
        ("College / University", "[ENTER COLLEGE NAME]"),
        ("Academic Year", "[ENTER ACADEMIC YEAR]"),
    ]

    for idx, (label, val) in enumerate(meta_items):
        p_l = tf_info.paragraphs[0] if idx == 0 else tf_info.add_paragraph()
        p_l.text = label.upper()
        p_l.font.name = FONT_TITLE
        p_l.font.size = Pt(8.5)
        p_l.font.bold = True
        p_l.font.color.rgb = TEAL if idx == 0 else RGBColor(148, 163, 184)
        p_l.space_after = Pt(1)

        p_v = tf_info.add_paragraph()
        p_v.text = val
        p_v.font.name = FONT_BODY
        p_v.font.size = Pt(11)
        p_v.font.bold = (idx > 0)
        p_v.font.color.rgb = RGBColor(255, 255, 255)
        p_v.space_after = Pt(8)

    # Footer note
    tb_foot = slide.shapes.add_textbox(Inches(0.9), Inches(6.5), Inches(7.5), Inches(0.4))
    tf_foot = tb_foot.text_frame
    p_f = tf_foot.paragraphs[0]
    p_f.text = "Python 3  |  Scikit-Learn  |  Streamlit  |  Pandas  |  NumPy  |  Plotly  |  Pytest"
    p_f.font.size = Pt(9.5)
    p_f.font.color.rgb = RGBColor(148, 163, 184)

    add_speaker_notes(slide,
        "Good morning respected evaluators, project guide, and faculty members. "
        "Welcome to our B.Tech project presentation titled 'Stock Return Prediction Using Ridge and Lasso Regression'. "
        "In this project, we have designed and implemented an end-to-end Machine Learning web application that allows users "
        "to upload any historical stock market CSV, automatically cleans the data, engineers leakage-free features, "
        "and compares two fundamental regularized regression techniques: Ridge and Lasso Regression."
    )


def build_slide_2_intro(prs):
    """SLIDE 2: Introduction."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Introduction to Stock Return Forecasting", "Project Context",
               "Understanding the distinction between non-stationary stock prices and stationary percentage returns.")
    add_footer(slide, 2)

    # Card 1: What Stock Data & Returns Represent
    add_card(slide, 0.8, 1.6, 5.7, 5.15)
    tb1 = slide.shapes.add_textbox(Inches(1.05), Inches(1.8), Inches(5.2), Inches(4.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Financial Market Data & Returns"
    p1.font.bold = True
    p1.font.size = Pt(16)
    p1.font.color.rgb = NAVY
    p1.space_after = Pt(10)

    points1 = [
        "Historical Stock Market Data: Comprises daily Open, High, Low, Close (OHLC) prices, Adjusted Close, and traded share Volume.",
        "Prices vs. Returns: Stock prices drift significantly over multi-decade horizons (non-stationary mean and variance), making raw price prediction unstable.",
        "Stationarity of Returns: Daily percentage returns reflect relative day-to-day percentage changes, centering stably near 0% across market cycles.",
        "Prediction Target: The objective is forecasting next-trading-day return (target_return = close_{t+1} / close_t - 1) using end-of-day information.",
    ]
    add_bullet_points(tf1, points1, pt_size=12, space_after=10, bold_first_n_words=1)

    # Card 2: Role of ML & Regularization
    add_card(slide, 6.8, 1.6, 5.7, 5.15)
    tb2 = slide.shapes.add_textbox(Inches(7.05), Inches(1.8), Inches(5.2), Inches(4.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Role of Machine Learning & Regularization"
    p2.font.bold = True
    p2.font.size = Pt(16)
    p2.font.color.rgb = NAVY
    p2.space_after = Pt(10)

    points2 = [
        "High Multicollinearity: Same-day OHLC prices share near-perfect correlation (r > 0.98), destabilizing standard Ordinary Least Squares (OLS).",
        "Ridge Regression (L2 Regularization): Shrinks regression weights smoothly toward zero, minimizing variance caused by collinear inputs.",
        "Lasso Regression (L1 Regularization): Enforces sparsity by setting non-informative or redundant feature weights to exact zero (feature selection).",
        "Automated Educational System: Provides a guided web application for students and analysts to upload CSVs, run automated preprocessing, and benchmark both models.",
    ]
    add_bullet_points(tf2, points2, pt_size=12, space_after=10, bold_first_n_words=1)

    add_speaker_notes(slide,
        "To introduce our project: In quantitative finance, predicting raw stock prices directly is a common beginner pitfall because prices drift "
        "and change scale across decades. Instead, our project predicts percentage returns, which remain stationary around zero. "
        "Furthermore, daily stock market attributes like Open, High, Low, and Close prices are strongly correlated with each other. "
        "Ordinary least squares regression fails in such conditions. This is where Ridge and Lasso regularized regression come in: "
        "Ridge controls coefficient inflation through L2 penalties, while Lasso performs automatic feature selection through L1 penalties."
    )


def build_slide_3_problem(prs):
    """SLIDE 3: Problem Statement."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Problem Statement & Analytical Challenges", "Problem Formulation",
               "Key technical challenges encountered in automated stock dataset processing and model evaluation.")
    add_footer(slide, 3)

    cards = [
        ("1. Data Inconsistency & Manual Overhead", [
            "Raw CSV files from diverse sources (Kaggle, Yahoo Finance, NSE) use inconsistent headers ('Date' vs 'Timestamp', 'Adj Close' vs 'Close').",
            "Manual data cleaning in spreadsheets is error-prone, tedious, and non-reproducible for recurring analysis.",
            "Missing observations, non-trading weekend records, and duplicate entries require robust automated normalization.",
        ]),
        ("2. Look-Ahead Bias & Data Leakage Pitfalls", [
            "A frequent error in academic ML projects is using random train-test splitting (e.g., standard train_test_split), which leaks future prices into past training.",
            "Feature scaling (StandardScaler) must be fitted strictly on the training partition only; leaking test statistics yields overly optimistic, false results.",
            "Target construction must strictly shift prices forward by exactly one session without leaking t+1 values into t features.",
        ]),
        ("3. High Market Noise & Model Evaluation", [
            "Daily stock returns exhibit a notoriously low signal-to-noise ratio and fat-tailed distributions (leptokurtosis).",
            "Linear models produce modest R² values; students need a tool that evaluates models honestly without fabricating unrealistic profit claims.",
            "Requires an integrated web interface that prevents premature prediction before models are properly trained on validated data.",
        ]),
    ]

    for idx, (title, points) in enumerate(cards):
        left = 0.8 + idx * 3.95
        add_card(slide, left, 1.6, 3.8, 5.15)
        tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(1.8), Inches(3.4), Inches(4.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = NAVY
        p.space_after = Pt(10)
        add_bullet_points(tf, points, pt_size=11, space_after=8, bold_first_n_words=1)

    add_speaker_notes(slide,
        "Here we articulate the core problem statement. Financial datasets present three major challenges: "
        "First, inconsistent formats across data providers make manual preprocessing repetitive and prone to error. "
        "Second, time-series data is vulnerable to look-ahead bias: if you randomly shuffle data or fit your scaler on the entire dataset, "
        "you leak future information into your model. "
        "Third, daily returns are noisy, meaning models must be evaluated strictly on chronologically held-out test data. "
        "Our system addresses all three challenges by enforcing a strict, automated, and leakage-free pipeline."
    )


def build_slide_4_motivation(prs):
    """SLIDE 4: Motivation."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Motivation Behind the Project", "Why This Matters",
               "Bridging classroom machine learning concepts with practical financial data science workflows.")
    add_footer(slide, 4)

    reasons = [
        ("Bridging ML Theory & Finance", [
            "Financial markets offer vast, accessible real-world data perfect for studying regression mechanics.",
            "Transforms abstract textbook concepts (L1 vs L2 regularization, multicollinearity, loss penalties) into tangible visual demonstrations.",
            "Teaches the necessity of statistical stationarity when working with time-dependent attributes.",
        ]),
        ("Pedagogical Value of Model Comparison", [
            "Direct empirical comparison between Ridge (smooth weight shrinkage) and Lasso (exact zero weight feature selection).",
            "Allows students and examiners to observe firsthand how Lasso naturally drops non-stationary price columns while retaining relative returns.",
            "Demystifies hyperparameter tuning (alpha) using chronological TimeSeriesSplit cross-validation.",
        ]),
        ("Need for an Accessible Analytical Tool", [
            "Removes the coding barrier for introductory data science students and researchers by providing an intuitive web interface.",
            "Fosters ethical, realistic AI education: explicitly teaches model limitations rather than promoting unrealistic get-rich-quick claims.",
            "Provides an end-to-end template for reproducible B.Tech engineering projects from CSV ingestion to exportable reports.",
        ]),
    ]

    for idx, (title, points) in enumerate(reasons):
        left = 0.8 + idx * 3.95
        add_card(slide, left, 1.6, 3.8, 5.15)
        tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(1.8), Inches(3.4), Inches(4.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = ROYAL_BLUE
        p.space_after = Pt(10)
        add_bullet_points(tf, points, pt_size=11, space_after=8, bold_first_n_words=1)

    add_speaker_notes(slide,
        "Our motivation stems from the desire to create an end-to-end, scientifically sound learning tool. "
        "Too often, stock prediction tutorials use toy setups with severe leakage and unrealistic claims. "
        "We wanted to build an application that teaches sound data science principles: "
        "how L1 and L2 regularization behave when presented with collinear financial features, "
        "how to properly evaluate time-series models, and how to communicate results transparently through a modern web UI."
    )


def build_slide_5_objectives(prs):
    """SLIDE 5: Objectives."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Project Objectives", "Goals & Deliverables",
               "Five clear, measurable engineering goals established for this project.")
    add_footer(slide, 5)

    objectives = [
        ("1. Dynamic CSV Upload & Schema Validation",
         "Enable users to upload any standard historical stock-market CSV file, automatically mapping common column name variations (Date, Open, High, Low, Close, Volume, Adj Close) with robust error checks."),
        ("2. Automated & Leakage-Free Preprocessing",
         "Clean uploaded datasets by parsing dates, sorting chronologically, removing duplicates and invalid OHLC prices without overwriting raw data or fabricating artificial stock values."),
        ("3. Comprehensive Exploratory Data Analysis (EDA)",
         "Generate 7 automated financial charts (price trends, volume moving averages, return distributions, correlation heatmaps, boxplots) with interactive date filtering and PNG exports."),
        ("4. Pipeline-Based Regularized Model Training",
         "Implement scikit-learn Pipelines combining StandardScaler with Ridge (L2) and Lasso (L1) regression, tuned strictly via chronological TimeSeriesSplit cross-validation on an 80% training split."),
        ("5. Objective Evaluation & Prediction Interface",
         "Evaluate both models on the unseen 20% test partition (MAE, MSE, RMSE, R²), provide dual-mode next-day predictions (Mode A: Historical & Mode B: Manual), and enable one-click report downloads."),
    ]

    for idx, (title, desc) in enumerate(objectives):
        top = 1.6 + idx * 1.05
        # Left icon/number card
        add_card(slide, 0.8, top, 0.8, 0.9, bg_color=NAVY, border_color=None)
        tb_num = slide.shapes.add_textbox(Inches(0.8), Inches(top + 0.2), Inches(0.8), Inches(0.5))
        p_n = tb_num.text_frame.paragraphs[0]
        p_n.text = f"0{idx+1}"
        p_n.font.name = FONT_TITLE
        p_n.font.size = Pt(16)
        p_n.font.bold = True
        p_n.font.color.rgb = RGBColor(255, 255, 255)
        p_n.alignment = PP_ALIGN.CENTER

        # Content card
        add_card(slide, 1.7, top, 10.8, 0.9, bg_color=CARD_WHITE, border_color=SLATE_BORDER)
        tb_c = slide.shapes.add_textbox(Inches(1.85), Inches(top + 0.08), Inches(10.5), Inches(0.75))
        tf_c = tb_c.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = tf_c.margin_top = 0

        p_t = tf_c.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(13)
        p_t.font.color.rgb = NAVY
        p_t.space_after = Pt(2)

        p_d = tf_c.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = SLATE_DARK

    add_speaker_notes(slide,
        "We defined five concrete objectives for this project: "
        "First, allow flexible CSV uploads with automatic column header mapping. "
        "Second, automate data cleaning and validation while preserving raw data integrity. "
        "Third, generate dynamic exploratory data visualizations with date range filtering. "
        "Fourth, train Ridge and Lasso regression inside leakage-free scikit-learn pipelines with time-series hyperparameter tuning. "
        "And fifth, evaluate models on unseen data and provide dual-mode predictions alongside downloadable CSV and Markdown reports."
    )


def build_slide_6_existing(prs):
    """SLIDE 6: Existing System."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Existing Systems & Traditional Approaches", "Current State of Practice",
               "Examining conventional methods of financial data analysis and their inherent limitations.")
    add_footer(slide, 6)

    # Left: Existing Approaches
    add_card(slide, 0.8, 1.6, 5.7, 5.15)
    tb1 = slide.shapes.add_textbox(Inches(1.05), Inches(1.8), Inches(5.2), Inches(4.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Common Conventional Approaches"
    p1.font.bold = True
    p1.font.size = Pt(16)
    p1.font.color.rgb = NAVY
    p1.space_after = Pt(10)

    points1 = [
        "Spreadsheet-Based Analysis: Analysts frequently use Excel or Google Sheets for manual formula calculations, moving averages, and basic charts.",
        "Disconnected Python Scripts: Separate ad-hoc scripts written for data loading, another for plotting, and another for modeling, lacking a unified interface.",
        "Classical Econometric Models: Traditional time-series modeling (e.g., ARIMA) requires manual order identification (p, d, q) and assumes strict linear Gaussian noise.",
        "Code-Heavy ML Notebooks: Jupyter notebooks that require manual code modifications whenever a new CSV dataset or ticker is analyzed.",
    ]
    add_bullet_points(tf1, points1, pt_size=12, space_after=10, bold_first_n_words=1)

    # Right: Limitations
    add_card(slide, 6.8, 1.6, 5.7, 5.15)
    tb2 = slide.shapes.add_textbox(Inches(7.05), Inches(1.8), Inches(5.2), Inches(4.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Key Limitations & Pitfalls"
    p2.font.bold = True
    p2.font.size = Pt(16)
    p2.font.color.rgb = RED_ACCENT
    p2.space_after = Pt(10)

    points2 = [
        "Repetitive Manual Effort: Analysts must rewrite preprocessing scripts for every new dataset format or column naming convention.",
        "High Risk of Data Leakage: Standard train_test_split functions randomly shuffle data, introducing look-ahead bias and invalidating test results.",
        "Lack of Fair Benchmarking: Models are rarely evaluated on identical chronological splits with identical feature engineering pipelines.",
        "Accessibility Barrier: Non-technical users cannot readily upload a dataset, inspect data quality, or generate verified predictions.",
    ]
    add_bullet_points(tf2, points2, pt_size=12, space_after=10, bold_first_n_words=1)

    add_speaker_notes(slide,
        "Looking at the existing landscape, most financial data exploration is done either in spreadsheets or in disconnected scripts. "
        "Spreadsheet analysis is manual, tedious, and cannot easily support machine learning pipelines. "
        "On the other hand, standalone Python scripts require coding knowledge and are prone to data leakage—such as random shuffling of time-series data. "
        "Our project overcomes these limitations by integrating data cleaning, leakage-free pipelines, and model comparison into a single web application."
    )


def build_slide_7_proposed(prs):
    """SLIDE 7: Proposed System."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Proposed System Architecture & Capabilities", "Our Solution",
               "An integrated, upload-driven machine learning web application built with Streamlit and Scikit-learn.")
    add_footer(slide, 7)

    # Feature Grid (2 columns x 3 rows)
    features = [
        ("📁 Dynamic CSV Upload & Normalization",
         "Accepts any stock CSV, automatically maps variations (e.g., 'Trade Date' -> 'date', 'Adjusted Close' -> 'adj_close'), and checks data integrity.",
         "✅ Implemented & Verified"),
        ("🧹 Automated In-Memory Preprocessing",
         "Removes duplicates, missing values, non-positive prices, and invalid OHLC bounds while preserving original raw data in session state.",
         "✅ Implemented & Verified"),
        ("📊 7 Automated Exploratory Visualizations",
         "Generates price trends, volume moving averages, return distributions, correlation heatmaps, and boxplots with date range filtering and PNG downloads.",
         "✅ Implemented & Verified"),
        ("⚙️ Leakage-Free Pipeline Training",
         "Combines StandardScaler with Ridge and Lasso inside scikit-learn Pipelines, tuned via 5-fold TimeSeriesSplit cross-validation on an 80% train split.",
         "✅ Implemented & Verified"),
        ("📏 Fair Out-of-Sample Evaluation",
         "Evaluates both models on the held-out 20% test partition using MAE, MSE, RMSE, and R² with side-by-side bar charts and actual vs predicted plots.",
         "✅ Implemented & Verified"),
        ("🔮 Dual-Mode Prediction & Reports",
         "Mode A predicts next trading day from latest historical data; Mode B accepts custom inputs with strict validation. Exports CSVs and Markdown reports.",
         "✅ Implemented & Verified"),
    ]

    for idx, (title, desc, status) in enumerate(features):
        col_idx = idx % 2
        row_idx = idx // 2
        left = 0.8 + col_idx * 5.95
        top = 1.6 + row_idx * 1.75

        add_card(slide, left, top, 5.75, 1.6, bg_color=CARD_WHITE, border_color=SLATE_BORDER)
        tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.12), Inches(5.35), Inches(1.35))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = 0

        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(13)
        p_t.font.color.rgb = NAVY

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = SLATE_DARK
        p_d.space_after = Pt(2)

        p_s = tf.add_paragraph()
        p_s.text = status
        p_s.font.size = Pt(9.5)
        p_s.font.bold = True
        p_s.font.color.rgb = TEAL

    add_speaker_notes(slide,
        "Here is our proposed system. It is structured into an interactive, 11-step web application. "
        "A user simply uploads their raw CSV. The application validates the columns, cleans the records, "
        "generates exploratory data analysis charts, engineers 10 leakage-free features, trains Ridge and Lasso models, "
        "evaluates them on chronological test data, and allows users to obtain next-day return predictions. "
        "All features listed here have been fully implemented, tested, and verified."
    )


def build_slide_8_architecture(prs):
    """SLIDE 8: System Architecture / Block Diagram."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "System Architecture & Block Diagram", "Technical Structure",
               "Visual block diagram showing end-to-end data flow from user upload to prediction and report export.")
    add_footer(slide, 8)

    # Draw Architecture Flow using PowerPoint Shapes
    # Row 1: User -> Streamlit -> Upload & Validation -> Preprocessing
    nodes_row1 = [
        ("USER", "Client Browser", 0.8, 1.7, 2.0, 0.9, ROYAL_BLUE),
        ("STREAMLIT UI", "app.py Dashboard", 3.2, 1.7, 2.2, 0.9, NAVY),
        ("CSV UPLOAD & VALIDATION", "data_loader.py", 5.8, 1.7, 2.6, 0.9, NAVY),
        ("DATA PREPROCESSING", "preprocessing.py", 8.8, 1.7, 2.5, 0.9, NAVY),
    ]

    for title, sub, l, t, w, h, col in nodes_row1:
        card = add_card(slide, l, t, w, h, bg_color=col, border_color=None, corner_radius=True)
        tb = slide.shapes.add_textbox(Inches(l), Inches(t + 0.15), Inches(w), Inches(h - 0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(11)
        p1.font.color.rgb = RGBColor(255, 255, 255)
        p1.alignment = PP_ALIGN.CENTER
        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = RGBColor(203, 213, 225)
        p2.alignment = PP_ALIGN.CENTER

    # Row 2: EDA <-> Feature Engineering -> Chronological Split (80/20)
    nodes_row2 = [
        ("EXPLORATORY DATA ANALYSIS", "eda.py (7 Charts)", 0.8, 3.1, 2.7, 0.9, TEAL),
        ("FEATURE ENGINEERING", "feature_engineering.py (10 Feats)", 4.0, 3.1, 2.9, 0.9, NAVY),
        ("CHRONOLOGICAL SPLIT", "Train: 80%  |  Test: 20%", 7.4, 3.1, 2.7, 0.9, NAVY),
    ]

    for title, sub, l, t, w, h, col in nodes_row2:
        card = add_card(slide, l, t, w, h, bg_color=col, border_color=None, corner_radius=True)
        tb = slide.shapes.add_textbox(Inches(l), Inches(t + 0.15), Inches(w), Inches(h - 0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(11)
        p1.font.color.rgb = RGBColor(255, 255, 255)
        p1.alignment = PP_ALIGN.CENTER
        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = RGBColor(203, 213, 225)
        p2.alignment = PP_ALIGN.CENTER

    # Row 3: Parallel Training (Ridge vs Lasso)
    nodes_row3 = [
        ("MODEL A: RIDGE REGRESSION", "StandardScaler + Ridge (L2)", 1.5, 4.4, 3.5, 0.9, ROYAL_BLUE),
        ("MODEL B: LASSO REGRESSION", "StandardScaler + Lasso (L1)", 5.5, 4.4, 3.5, 0.9, RED_ACCENT),
    ]

    for title, sub, l, t, w, h, col in nodes_row3:
        card = add_card(slide, l, t, w, h, bg_color=col, border_color=None, corner_radius=True)
        tb = slide.shapes.add_textbox(Inches(l), Inches(t + 0.15), Inches(w), Inches(h - 0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(11)
        p1.font.color.rgb = RGBColor(255, 255, 255)
        p1.alignment = PP_ALIGN.CENTER
        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = RGBColor(203, 213, 225)
        p2.alignment = PP_ALIGN.CENTER

    # Row 4: Evaluation -> Prediction -> Export
    nodes_row4 = [
        ("MODEL EVALUATION", "MAE, MSE, RMSE, R²", 0.8, 5.7, 2.7, 0.9, NAVY),
        ("MODEL COMPARISON", "Actual vs Predicted Plot", 3.8, 5.7, 2.7, 0.9, NAVY),
        ("RETURN PREDICTION", "Mode A & Mode B", 6.8, 5.7, 2.4, 0.9, ROYAL_BLUE),
        ("REPORT DOWNLOADS", "report.py (CSV/PNG/MD)", 9.5, 5.7, 2.7, 0.9, TEAL),
    ]

    for title, sub, l, t, w, h, col in nodes_row4:
        card = add_card(slide, l, t, w, h, bg_color=col, border_color=None, corner_radius=True)
        tb = slide.shapes.add_textbox(Inches(l), Inches(t + 0.15), Inches(w), Inches(h - 0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = RGBColor(255, 255, 255)
        p1.alignment = PP_ALIGN.CENTER
        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = RGBColor(203, 213, 225)
        p2.alignment = PP_ALIGN.CENTER

    add_speaker_notes(slide,
        "This slide presents the architectural block diagram of our system. "
        "Notice the clear modular separation: "
        "The user interacts via Streamlit. Data moves through upload validation, preprocessing, and EDA. "
        "Feature engineering constructs 10 inputs and the target. "
        "The chronological split feeds both the Ridge pipeline and Lasso pipeline in parallel. "
        "Outputs are combined for objective metric comparison and prediction generation, with full download capability."
    )


def build_slide_9_methodology(prs):
    """SLIDE 9: Methodology / Workflow."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Methodology & End-to-End Workflow", "System Workflow",
               "Step-by-step execution path ensuring chronological integrity and zero future data leakage.")
    add_footer(slide, 9)

    steps = [
        ("Step 1: Dataset Upload", "User uploads a raw stock CSV or selects the bundled Reliance dataset via Streamlit uploader."),
        ("Step 2: Input Validation", "Validates required OHLCV columns, checks date format, and computes dataset hash for state tracking."),
        ("Step 3: Preprocessing", "Parses dates, removes duplicate dates, filters non-positive prices, and sorts chronologically."),
        ("Step 4: Exploratory Analysis", "Generates 7 charts (price trend, volume MA, return distributions, heatmaps, boxplots) with date filters."),
        ("Step 5: Feature Engineering", "Computes daily return, intraday return, daily range, lag return, and 5-day return. Target = next day return."),
        ("Step 6: Chronological Split", "Splits the first 80% of rows for training and the final 20% for testing. Shuffling is strictly prohibited."),
        ("Step 7: Regularized Training", "Fits StandardScaler only on training data; tunes alpha via 5-fold TimeSeriesSplit cross-validation."),
        ("Step 8: Model Evaluation", "Calculates MAE, MSE, RMSE, and R² on the unseen 20% test partition without retraining."),
        ("Step 9: Prediction & Reports", "Mode A (latest historical) & Mode B (manual input). Generates downloadable CSVs, PNGs, and Markdown reports."),
    ]

    for idx, (title, desc) in enumerate(steps):
        col_idx = idx % 3
        row_idx = idx // 3
        left = 0.8 + col_idx * 3.95
        top = 1.6 + row_idx * 1.75

        add_card(slide, left, top, 3.8, 1.6, bg_color=CARD_WHITE, border_color=SLATE_BORDER)
        tb = slide.shapes.add_textbox(Inches(left + 0.15), Inches(top + 0.12), Inches(3.5), Inches(1.35))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = 0

        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(12.5)
        p_t.font.color.rgb = ROYAL_BLUE
        p_t.space_after = Pt(3)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = SLATE_DARK

    add_speaker_notes(slide,
        "Here is the detailed 9-step methodology. "
        "A critical principle enforced throughout is zero data leakage: "
        "all feature engineering uses information available at or before the close of trading day t. "
        "The scaler is fitted only on the training 80% split, and hyperparameter tuning uses TimeSeriesSplit, "
        "ensuring validation folds never look into future periods."
    )


def build_slide_10_tech_stack(prs):
    """SLIDE 10: Technology Stack."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Technology Stack & Software Environment", "Tools & Frameworks",
               "Core open-source libraries and frameworks used to implement the application.")
    add_footer(slide, 10)

    techs = [
        ("Python 3.9+", "Core Language", "Primary programming language chosen for its rich ecosystem in scientific computing and machine learning."),
        ("Streamlit 1.50", "Web Interface", "Delivers an interactive, reactive, multi-page financial dashboard with progress tracking and file uploaders."),
        ("Pandas 2.3", "Data Processing", "Handles CSV ingestion, timezone normalization, chronological sorting, rolling windows, and feature computation."),
        ("NumPy 2.0", "Numerical Computing", "High-performance vectorized array math for return calculations and directional accuracy checks."),
        ("Scikit-Learn 1.6", "Machine Learning", "Provides StandardScaler, Ridge, Lasso, Pipeline, TimeSeriesSplit, GridSearchCV, and regression metrics."),
        ("Plotly & Seaborn", "Visualization", "Plotly generates interactive zoomable web charts; Matplotlib & Seaborn generate static high-res PNGs."),
        ("Joblib 1.5", "Model Persistence", "Serializes and loads trained scikit-learn Pipeline artifacts to disk without retraining on every interaction."),
        ("Pytest 8.4", "Automated Testing", "Test suite containing 31 automated unit and integration tests verifying preprocessing, leakage, and models."),
    ]

    for idx, (name, role, desc) in enumerate(techs):
        col_idx = idx % 4
        row_idx = idx // 4
        left = 0.8 + col_idx * 2.95
        top = 1.6 + row_idx * 2.65

        add_card(slide, left, top, 2.8, 2.45, bg_color=CARD_WHITE, border_color=SLATE_BORDER)
        tb = slide.shapes.add_textbox(Inches(left + 0.15), Inches(top + 0.15), Inches(2.5), Inches(2.15))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = 0

        p_n = tf.paragraphs[0]
        p_n.text = name
        p_n.font.bold = True
        p_n.font.size = Pt(14)
        p_n.font.color.rgb = NAVY
        p_n.space_after = Pt(2)

        p_r = tf.add_paragraph()
        p_r.text = role.upper()
        p_r.font.bold = True
        p_r.font.size = Pt(9)
        p_r.font.color.rgb = TEAL
        p_r.space_after = Pt(6)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = SLATE_DARK

    add_speaker_notes(slide,
        "Our technology stack relies strictly on standard, verified open-source libraries. "
        "Python is the core language; Streamlit provides the reactive frontend; Pandas and NumPy handle data transformations; "
        "Scikit-learn implements Ridge and Lasso pipelines; Plotly and Matplotlib deliver interactive and static visualizations; "
        "Joblib serializes model artifacts; and Pytest ensures code reliability through 31 automated tests."
    )


def build_slide_11_modules(prs):
    """SLIDE 11: System Modules / Components."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "System Modules & Architecture Breakdown", "Component Design",
               "Eight modular Python components with clear separation of concerns, inputs, and outputs.")
    add_footer(slide, 11)

    headers = ["Module Name", "Source File", "Primary Purpose", "Inputs / Outputs"]
    rows = [
        ["1. User Interface Module", "app.py", "Renders the 9-page Streamlit web app, progress bar, and user controls.", "User input / Reactive dashboard UI"],
        ["2. CSV Upload & Validation", "src/data_loader.py", "Normalizes column variations, validates OHLCV schema, checks date validity.", "Raw CSV file -> Validated DataFrame"],
        ["3. Data Preprocessing", "src/preprocessing.py", "Cleans data, removes duplicates, handles invalid prices, sorts chronologically.", "Raw DataFrame -> Cleaned DataFrame"],
        ["4. Exploratory Data Analysis", "src/eda.py", "Generates 7 statistical visualizations with interpretations and PNG downloads.", "Cleaned DataFrame -> Static & Plotly charts"],
        ["5. Feature Engineering", "src/feature_engineering.py", "Constructs 10 leakage-free features and calculates next-day target return.", "Cleaned DataFrame -> (X, y) Modeling Data"],
        ["6. Model Training Module", "src/train.py", "Splits data 80/20 chronologically; tunes Ridge & Lasso via TimeSeriesSplit CV.", "Modeling Data -> Fitted Pipeline models"],
        ["7. Evaluation & Comparison", "src/evaluate.py", "Calculates MAE, MSE, RMSE, R², and directional accuracy on unseen test set.", "Test Split + Models -> Metrics Table & Plots"],
        ["8. Prediction & Reporting", "src/predict.py & report.py", "Inference via Mode A & Mode B; generates Markdown and text summary reports.", "Inputs + Models -> Predictions & Reports"],
    ]

    add_table(slide, 0.8, 1.6, 11.733, 5.15, headers, rows, col_widths=[2.4, 2.0, 4.333, 3.0])

    add_speaker_notes(slide,
        "Here we show the breakdown of the eight modular components in our codebase. "
        "Each module has a strict single responsibility: data loading in data_loader.py, "
        "cleaning in preprocessing.py, feature calculation in feature_engineering.py, "
        "training in train.py, evaluation in evaluate.py, and prediction/reporting in predict.py and report.py. "
        "This makes the codebase clean, maintainable, and thoroughly testable."
    )


def build_slide_12_implementation(prs):
    """SLIDE 12: Implementation."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Implementation Details & Code Architecture", "Code Implementation",
               "Key Python implementation strategies, scikit-learn pipelines, and target formulation.")
    add_footer(slide, 12)

    # Left Card: Target & Feature Formulation
    add_card(slide, 0.8, 1.6, 5.7, 5.15)
    tb1 = slide.shapes.add_textbox(Inches(1.05), Inches(1.8), Inches(5.2), Inches(4.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Target Variable & Feature Implementation"
    p1.font.bold = True
    p1.font.size = Pt(15)
    p1.font.color.rgb = NAVY
    p1.space_after = Pt(8)

    p_code1 = tf1.add_paragraph()
    p_code1.text = "# Next-Trading-Day Return Formulation (day t+1):\ndf['target_return'] = df['close'].shift(-1) / df['close'] - 1.0\n# Drop final observation (target is unobserved):\ndf = df.dropna(subset=['target_return'])"
    p_code1.font.name = "Courier New"
    p_code1.font.size = Pt(9.5)
    p_code1.font.color.rgb = NAVY
    p_code1.space_after = Pt(10)

    points1 = [
        "Chronological Target: Shift(-1) divides tomorrow's close by today's close, subtracting 1 to obtain percentage change.",
        "Warmup Dropping: The 5-day lag feature (return_5d) introduces 5 initial NaN rows; these are cleanly dropped before training.",
        "10 Inputs at Day t: OHLC prices, volume, daily_return, intraday_return, daily_range, lag_return_1, and return_5d.",
        "Zero Look-Ahead Bias: target_return is never included inside input features X.",
    ]
    add_bullet_points(tf1, points1, pt_size=11, space_after=6, bold_first_n_words=1)

    # Right Card: Scikit-learn Pipeline
    add_card(slide, 6.8, 1.6, 5.7, 5.15)
    tb2 = slide.shapes.add_textbox(Inches(7.05), Inches(1.8), Inches(5.2), Inches(4.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Pipeline Architecture & Tuning"
    p2.font.bold = True
    p2.font.size = Pt(15)
    p2.font.color.rgb = NAVY
    p2.space_after = Pt(8)

    p_code2 = tf2.add_paragraph()
    p_code2.text = "# Leakage-Free Scikit-Learn Pipeline:\nridge_pipeline = Pipeline([\n    ('scaler', StandardScaler()),\n    ('regressor', Ridge(alpha=best_alpha))\n])\n# TimeSeriesSplit Hyperparameter Tuning:\ntscv = TimeSeriesSplit(n_splits=5)\ngrid = GridSearchCV(ridge_pipeline, param_grid, cv=tscv)"
    p_code2.font.name = "Courier New"
    p_code2.font.size = Pt(9.5)
    p_code2.font.color.rgb = ROYAL_BLUE
    p_code2.space_after = Pt(10)

    points2 = [
        "Encapsulated Preprocessing: StandardScaler is combined directly with the regressor inside Pipeline, preventing data leakage.",
        "Fit Exclusively on Train: Scaler statistics (mean, variance) are learned solely from the first 80% of chronological observations.",
        "Cross-Validation: TimeSeriesSplit enforces temporal order across 5 folds so past folds never train on future folds.",
        "Model Persistence: Pipelines are serialized using joblib.dump for instant zero-latency loading in Streamlit.",
    ]
    add_bullet_points(tf2, points2, pt_size=11, space_after=6, bold_first_n_words=1)

    add_speaker_notes(slide,
        "Here are the core implementation details in Python. "
        "The target is created with close.shift(-1) divided by close minus 1. "
        "The final row is dropped because tomorrow's price is unobserved. "
        "On the right, we show how we implement scikit-learn Pipelines: StandardScaler and Ridge or Lasso are bundled together. "
        "This guarantees that test data is never seen during scaling, preventing subtle data leakage."
    )


def build_slide_13_dataset(prs):
    """SLIDE 13: Dataset / Input Data."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Dataset Schema & Input Characteristics", "Data Description",
               "Initial examination using Kaggle Reliance Industries stock dataset, with dynamic support for custom CSVs.")
    add_footer(slide, 13)

    # Top summary metrics
    top_metrics = [
        ("Reference Dataset", "Reliance Industries (RELIANCE.NS)", "Source: Kaggle (1996 - 2025)"),
        ("Raw Observations", "7,324 Trading Days", "0 missing, 0 duplicates"),
        ("Modeling Observations", "7,318 Rows", "After 5 lag warmup + 1 target drop"),
        ("Date Span", "Jan 1, 1996 - Feb 28, 2025", "29 years of market history"),
    ]

    for idx, (lbl, val, sub) in enumerate(top_metrics):
        left = 0.8 + idx * 2.95
        add_card(slide, left, 1.6, 2.8, 1.05, bg_color=CARD_WHITE, border_color=SLATE_BORDER)
        tb = slide.shapes.add_textbox(Inches(left + 0.1), Inches(1.68), Inches(2.6), Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = 0
        p1 = tf.paragraphs[0]
        p1.text = lbl.upper()
        p1.font.bold = True
        p1.font.size = Pt(8.5)
        p1.font.color.rgb = TEAL
        p2 = tf.add_paragraph()
        p2.text = val
        p2.font.bold = True
        p2.font.size = Pt(12)
        p2.font.color.rgb = NAVY
        p3 = tf.add_paragraph()
        p3.text = sub
        p3.font.size = Pt(8.5)
        p3.font.color.rgb = SLATE_MUTED

    # Table of Schema Columns
    headers = ["Column Name", "Data Type", "Role in Pipeline", "Financial Definition / Meaning"]
    rows = [
        ["date", "Datetime64", "Index / Chronology", "Trading session date (normalized to timezone-naive format)."],
        ["open", "Float64 (INR)", "Input Feature", "Stock price at market open on trading day t."],
        ["high", "Float64 (INR)", "Input Feature", "Highest price reached during trading day t."],
        ["low", "Float64 (INR)", "Input Feature", "Lowest price reached during trading day t."],
        ["close", "Float64 (INR)", "Input Feature / Base", "Closing price at end of day t (used as denominator for return)."],
        ["adj_close", "Float64 (INR)", "Optional Reference", "Close adjusted for stock splits and dividends (treated as optional)."],
        ["volume", "Int64", "Input Feature", "Total number of shares traded during trading day t."],
        ["target_return", "Float64", "Prediction Target", "Next-trading-day return: (close_{t+1} / close_t) - 1. Removed from inputs."],
    ]

    add_table(slide, 0.8, 2.85, 11.733, 3.9, headers, rows, col_widths=[1.8, 1.5, 2.2, 6.233])

    add_speaker_notes(slide,
        "Here is the dataset schema. We initially examined the Kaggle Reliance Industries stock dataset, covering 7,324 days from 1996 to 2025. "
        "The table displays the columns: Date, Open, High, Low, Close, Adjusted Close, and Volume. "
        "Importantly, our web application is designed to ingest any compatible CSV file—the Reliance dataset serves as our verified reference, "
        "not a hard-coded constraint."
    )


def build_slide_14_algorithm(prs):
    """SLIDE 14: Algorithm / Model (Ridge vs. Lasso)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Algorithmic Comparison: Ridge vs. Lasso Regression", "Machine Learning Foundations",
               "Theoretical formulation of L2 vs. L1 penalties and their response to collinear financial features.")
    add_footer(slide, 14)

    # Left: Ridge Regression
    add_card(slide, 0.8, 1.6, 5.7, 5.15)
    tb1 = slide.shapes.add_textbox(Inches(1.05), Inches(1.8), Inches(5.2), Inches(4.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Ridge Regression (L2 Regularization)"
    p1.font.bold = True
    p1.font.size = Pt(16)
    p1.font.color.rgb = ROYAL_BLUE
    p1.space_after = Pt(6)

    p_eq1 = tf1.add_paragraph()
    p_eq1.text = "Objective Function:\nmin  (1 / 2N) * ||y - Xw||² + α * Σ(w_j²)"
    p_eq1.font.name = "Courier New"
    p_eq1.font.size = Pt(10)
    p_eq1.font.bold = True
    p_eq1.font.color.rgb = NAVY
    p_eq1.space_after = Pt(10)

    points1 = [
        "L2 Penalty: Penalizes the sum of squared coefficient weights (w²).",
        "Smooth Shrinkage: Shrinks collinear weights proportionately toward zero without setting any weight to exact zero.",
        "Retains All Features: Keeps all 10 inputs active in the prediction formula, distributing weight across correlated price features.",
        "Hyperparameter α: Tuned across [0.001 to 1000.0] via TimeSeriesSplit CV; best α = 10.0.",
    ]
    add_bullet_points(tf1, points1, pt_size=11, space_after=8, bold_first_n_words=1)

    # Right: Lasso Regression
    add_card(slide, 6.8, 1.6, 5.7, 5.15)
    tb2 = slide.shapes.add_textbox(Inches(7.05), Inches(1.8), Inches(5.2), Inches(4.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Lasso Regression (L1 Regularization)"
    p2.font.bold = True
    p2.font.size = Pt(16)
    p2.font.color.rgb = RED_ACCENT
    p2.space_after = Pt(6)

    p_eq2 = tf2.add_paragraph()
    p_eq2.text = "Objective Function:\nmin  (1 / 2N) * ||y - Xw||² + α * Σ|w_j|"
    p_eq2.font.name = "Courier New"
    p_eq2.font.size = Pt(10)
    p_eq2.font.bold = True
    p_eq2.font.color.rgb = NAVY
    p_eq2.space_after = Pt(10)

    points2 = [
        "L1 Penalty: Penalizes the sum of absolute coefficient weights (|w|).",
        "Sparsity & Feature Selection: The sharp diamond geometry of L1 drives redundant and collinear feature weights to exact zero.",
        "Zeroing Out Price Levels: Lasso automatically set open, high, low, close weights to 0.0, retaining only the 6 relative return/volume features.",
        "Hyperparameter α: Tuned across [1e-6 to 0.1] via TimeSeriesSplit CV; best α = 0.0005.",
    ]
    add_bullet_points(tf2, points2, pt_size=11, space_after=8, bold_first_n_words=1)

    add_speaker_notes(slide,
        "Here we compare the two algorithms mathematically. "
        "Ridge adds an L2 penalty on squared weights. It shrinks weights smoothly, helping when variables are correlated, but retains all features. "
        "Lasso adds an L1 penalty on absolute weights. Because the L1 constraint has sharp corners, it drives less useful weights to exact zero. "
        "In our project, this difference is crucial: Lasso automatically eliminated raw price levels, focusing purely on stationary return and volume features."
    )


def build_slide_15_results(prs):
    """SLIDE 15: Results & Performance Evaluation."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Model Evaluation & Out-of-Sample Performance", "Experimental Results",
               "Evaluated on 1,464 unseen test trading days (March 28, 2019 to February 27, 2025).")
    add_footer(slide, 15)

    # Left: Results Table + Key Findings
    add_card(slide, 0.8, 1.6, 6.2, 5.15)
    tb_t = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.8), Inches(4.7))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = "Test-Set Performance Comparison Table"
    p_t.font.bold = True
    p_t.font.size = Pt(14)
    p_t.font.color.rgb = NAVY
    p_t.space_after = Pt(8)

    # Embed table inside left side
    headers = ["Model", "MAE", "MSE", "RMSE", "R² Score", "Directional Acc."]
    rows = [
        ["Ridge Regression (L2)", "0.012671", "0.000346", "0.018595", "-0.024513", "49.18%"],
        ["Lasso Regression (L1)", "0.012516", "0.000340", "0.018439", "-0.007373", "49.66%"],
    ]
    add_table(slide, 1.0, 2.3, 5.8, 1.5, headers, rows, col_widths=[1.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    # Text interpretation below table
    tb_desc = slide.shapes.add_textbox(Inches(1.0), Inches(4.0), Inches(5.8), Inches(2.6))
    tf_desc = tb_desc.text_frame
    tf_desc.word_wrap = True
    points = [
        "Lasso Outperformed Ridge: Achieved lower MAE (0.012516 vs 0.012671), lower RMSE (0.018439 vs 0.018595), and higher R² (-0.007373 vs -0.024513).",
        "Why Lasso Won: Reliance stock price surged from ~Rs.500 (train period) to >Rs.1200 (test period). Lasso set open, high, low, close to 0.0, avoiding scale distortion.",
        "Honest Interpretation of Near-Zero R²: Daily stock returns are dominated by market noise. A slightly negative or near-zero out-of-sample R² is standard in honest finance ML.",
    ]
    add_bullet_points(tf_desc, points, pt_size=10.5, space_after=6, bold_first_n_words=1)

    # Right: Embed Actual Generated Comparison Bar Chart
    add_card(slide, 7.2, 1.6, 5.333, 5.15)
    chart_path = GRAPHS_DIR / "8_model_comparison_bar_chart.png"
    if chart_path.exists():
        slide.shapes.add_picture(str(chart_path), Inches(7.35), Inches(1.8), Inches(5.033), Inches(4.7))
    else:
        tb_p = slide.shapes.add_textbox(Inches(7.4), Inches(2.5), Inches(4.9), Inches(3.0))
        p = tb_p.text_frame.paragraphs[0]
        p.text = "[Model Comparison Bar Chart]\nRun python -m src.evaluate to view saved chart."
        p.alignment = PP_ALIGN.CENTER

    add_speaker_notes(slide,
        "Here are our actual out-of-sample results on 1,464 test trading days. "
        "Lasso Regression achieved lower MAE and lower RMSE than Ridge Regression. "
        "Why did Lasso win? Because nominal stock prices more than doubled between the training era (1996-2019) and the test era (2019-2025). "
        "Ridge kept all 10 features, which distorted its predictions under the higher price level. "
        "Lasso's L1 penalty automatically zeroed out the raw price levels, relying only on stationary return and volume features."
    )


def build_slide_16_demo(prs):
    """SLIDE 16: Screenshots / Demo."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Application Interface & Functional Demonstration", "Streamlit Web App",
               "Visual tour of the 9 guided pages implemented in the live interactive web application (app.py).")
    add_footer(slide, 16)

    demos = [
        ("1. Upload & Validation Page",
         "Features drag-and-drop CSV uploader, automatic column mapping (Date, Open, High, Low, Close, Volume), dataset hash verification, and 'Analyze Dataset' button.",
         TEAL),
        ("2. Dataset Analysis & Audit",
         "Presents 10 diagnostic metrics, missing value audit table, duplicate record count, date span check, and raw vs cleaned comparison.",
         NAVY),
        ("3. Exploratory Data Analysis",
         "Interactive Plotly & static Matplotlib charts: price trends, 30-day volume MA, return distributions, correlation heatmaps, and boxplots with date filtering.",
         ROYAL_BLUE),
        ("4. Feature & Model Training",
         "Displays the chronological 80/20 split and trains StandardScaler + Ridge/Lasso pipelines on demand via TimeSeriesSplit CV.",
         NAVY),
        ("5. Model Comparison & Metrics",
         "Side-by-side MAE/MSE/RMSE/R² comparison table, interactive error bar charts, actual vs predicted returns plot, and honest evaluation summary.",
         RED_ACCENT),
        ("6. Prediction & Report Downloads",
         "Mode A (latest historical data) & Mode B (manual input). Generates next-day return %, implied price, and exportable CSV/PNG/Markdown reports.",
         TEAL),
    ]

    for idx, (title, desc, accent) in enumerate(demos):
        col_idx = idx % 3
        row_idx = idx // 3
        left = 0.8 + col_idx * 3.95
        top = 1.6 + row_idx * 2.65

        card = add_card(slide, left, top, 3.8, 2.45, bg_color=CARD_WHITE, border_color=SLATE_BORDER)
        # Top color accent strip
        strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(3.8), Inches(0.08))
        strip.fill.solid()
        strip.fill.fore_color.rgb = accent
        strip.line.fill.background()

        tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.15), Inches(3.4), Inches(2.15))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = 0

        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(13)
        p_t.font.color.rgb = NAVY
        p_t.space_after = Pt(6)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = SLATE_DARK

    add_speaker_notes(slide,
        "Here we highlight the functional demonstration of the web application. "
        "The application is accessible locally at localhost:8501. "
        "It features an intuitive 8-stage progress tracker across 9 pages: from CSV ingestion and data health auditing, "
        "to interactive EDA, chronological pipeline training, side-by-side metric comparison, and dual-mode return forecasting. "
        "All predictions include our mandatory academic disclaimer."
    )


def build_slide_17_advantages(prs):
    """SLIDE 17: Advantages."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "System Advantages & Key Innovations", "Strengths",
               "Unique benefits of our automated, leakage-free financial machine learning platform.")
    add_footer(slide, 17)

    advantages = [
        ("End-to-End Automation",
         "Eliminates repetitive manual spreadsheet work by integrating CSV ingestion, normalization, cleaning, EDA, training, and prediction into a single responsive UI."),
        ("Strict Leakage Prevention",
         "Enforces chronological train-test splitting and pipeline-encapsulated scaling (StandardScaler), guaranteeing zero look-ahead bias."),
        ("Flexible Schema Mapping",
         "Automatically detects and maps common column header variations across market data providers, requiring zero manual script editing."),
        ("Rigorous Model Comparison",
         "Compares Ridge (L2) and Lasso (L1) on identical chronological partitions with TimeSeriesSplit tuning, demonstrating regularization effects clearly."),
        ("Dual-Mode Inference & Export",
         "Supports instant forecasting from latest historical data (Mode A) and custom scenario testing (Mode B), with one-click CSV and report downloads."),
        ("Honest Scientific Reporting",
         "Refrains from artificial profit claims, transparently explains near-zero R² values on noisy return data, and includes clear financial risk disclaimers."),
    ]

    for idx, (title, desc) in enumerate(advantages):
        col_idx = idx % 2
        row_idx = idx // 2
        left = 0.8 + col_idx * 5.95
        top = 1.6 + row_idx * 1.75

        add_card(slide, left, top, 5.75, 1.6, bg_color=CARD_WHITE, border_color=SLATE_BORDER)
        tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.12), Inches(5.35), Inches(1.35))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = 0

        p_t = tf.paragraphs[0]
        p_t.text = "✓ " + title
        p_t.font.bold = True
        p_t.font.size = Pt(13)
        p_t.font.color.rgb = TEAL
        p_t.space_after = Pt(3)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = SLATE_DARK

    add_speaker_notes(slide,
        "The primary advantages of our system are: "
        "first, complete workflow integration; "
        "second, guaranteed leakage prevention through chronological pipelines; "
        "third, flexible schema mapping; "
        "fourth, transparent and honest evaluation that teaches real data science rather than making unrealistic profit claims."
    )


def build_slide_18_applications(prs):
    """SLIDE 18: Applications."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Educational & Analytical Applications", "Use Cases",
               "Practical environments where this web application delivers substantial educational and analytical value.")
    add_footer(slide, 18)

    apps = [
        ("1. Academic Machine Learning Labs",
         "Serves as an interactive laboratory tool for B.Tech/M.Tech students to experiment with regression regularization, feature collinearity, and time-series evaluation.",
         ROYAL_BLUE),
        ("2. Quantitative Finance Seminars",
         "Demonstrates the vital empirical difference between non-stationary price forecasting and stationary return forecasting in university finance curricula.",
         NAVY),
        ("3. Algorithmic Baseline Benchmarking",
         "Provides a transparent, reproducible linear baseline benchmark against which complex deep learning (LSTM) or boosting models can be tested.",
         TEAL),
        ("4. Financial Data Quality Auditing",
         "Enables students and analysts to quickly inspect stock CSV health, spot price anomalies, check trading volume patterns, and identify duplicate dates.",
         NAVY),
    ]

    for idx, (title, desc, col) in enumerate(apps):
        col_idx = idx % 2
        row_idx = idx // 2
        left = 0.8 + col_idx * 5.95
        top = 1.6 + row_idx * 2.2

        add_card(slide, left, top, 5.75, 2.05, bg_color=CARD_WHITE, border_color=SLATE_BORDER)
        tb = slide.shapes.add_textbox(Inches(left + 0.25), Inches(top + 0.18), Inches(5.25), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = 0

        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(14)
        p_t.font.color.rgb = col
        p_t.space_after = Pt(6)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = SLATE_DARK

    # Disclaimer Callout Box
    add_card(slide, 0.8, 6.0, 11.733, 0.8, bg_color=RGBColor(254, 243, 199), border_color=RGBColor(245, 158, 11))
    tb_disc = slide.shapes.add_textbox(Inches(1.0), Inches(6.08), Inches(11.333), Inches(0.65))
    p_disc = tb_disc.text_frame.paragraphs[0]
    p_disc.text = "⚠️ Mandatory Educational Disclaimer: All predictions and metrics are experimental estimates designed for academic study. The application does not guarantee future market movements and does not constitute personalized investment advice."
    p_disc.font.size = Pt(10)
    p_disc.font.bold = True
    p_disc.font.color.rgb = RGBColor(146, 64, 14)

    add_speaker_notes(slide,
        "Regarding real-world applications: This system is primarily designed for academic laboratories, quantitative finance coursework, "
        "and baseline algorithm benchmarking. It provides a transparent, interactive testbed. "
        "We emphasize that all outputs are educational and do not constitute financial advice."
    )


def build_slide_19_limitations(prs):
    """SLIDE 19: Limitations."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Project Limitations & Market Realities", "Critical Analysis",
               "Acknowledging empirical constraints, linear assumptions, and unmodeled real-world financial factors.")
    add_footer(slide, 19)

    limits = [
        ("High Noise-to-Signal Ratio", [
            "Daily stock returns are heavily dominated by random market noise and unforeseen breaking news (earnings releases, RBI policy shifts, geopolitical events).",
            "Consistent with the Efficient Market Hypothesis (EMH), historical price patterns alone explain only a small fraction of next-day return variance.",
        ]),
        ("Linearity Constraint of Ridge & Lasso", [
            "Both models assume strictly linear relationships between engineered technical features and the target return.",
            "Cannot capture complex non-linear interactions, regime shifts (bull vs bear markets), or asymmetric volatility without manual polynomial expansion.",
        ]),
        ("Absence of Market Frictions", [
            "Predictions reflect theoretical gross returns and do not account for brokerage charges, Securities Transaction Tax (STT), or bid-ask slippage.",
            "A small positive theoretical return may not translate into profitable real-world trading after accounting for execution costs.",
        ]),
        ("Corporate Actions & Adjustments", [
            "Stock splits, bonus share issues, and dividends require adjusted closing prices (adj_close); unadjusted datasets may show artificial price drops.",
            "Long-term nominal price drift can distort models that retain raw price levels (which is precisely why Lasso outperformed Ridge).",
        ]),
    ]

    for idx, (title, points) in enumerate(limits):
        col_idx = idx % 2
        row_idx = idx // 2
        left = 0.8 + col_idx * 5.95
        top = 1.6 + row_idx * 2.65

        add_card(slide, left, top, 5.75, 2.45, bg_color=CARD_WHITE, border_color=SLATE_BORDER)
        tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.15), Inches(5.35), Inches(2.15))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = 0

        p_t = tf.paragraphs[0]
        p_t.text = "⚠️ " + title
        p_t.font.bold = True
        p_t.font.size = Pt(13.5)
        p_t.font.color.rgb = RED_ACCENT
        p_t.space_after = Pt(6)

        add_bullet_points(tf, points, pt_size=10.5, space_after=6, bold_first_n_words=1)

    add_speaker_notes(slide,
        "A rigorous engineering project must acknowledge its limitations. "
        "First, daily stock returns have a low signal-to-noise ratio—news and sentiment drive prices in ways historical OHLCV data cannot foresee. "
        "Second, Ridge and Lasso are linear models and cannot capture non-linear market regimes. "
        "Third, transaction costs and slippage are not modeled. "
        "Recognizing these boundaries demonstrates sound scientific understanding."
    )


def build_slide_20_future(prs):
    """SLIDE 20: Future Scope."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Future Scope & Technical Enhancements", "Roadmap Ahead",
               "Potential future extensions to expand modeling capability, data sources, and deployment architecture.")
    add_footer(slide, 20)

    scopes = [
        ("1. Non-Linear & Ensemble Algorithms",
         "Integrate Random Forests, Gradient Boosted Decision Trees (XGBoost/LightGBM), and Long Short-Term Memory (LSTM) recurrent neural networks to capture non-linear temporal dynamics."),
        ("2. Alternative Data & Sentiment NLP",
         "Incorporate sentiment analysis scores extracted from financial news headlines, Twitter/X sentiment feeds, and quarterly earnings call transcripts via NLP transformer models."),
        ("3. Macroeconomic & Benchmark Covariates",
         "Expand the feature matrix to include Nifty 50 benchmark returns, Brent crude oil prices, INR/USD exchange rates, and RBI 10-year bond yields."),
        ("4. Rolling Walk-Forward Retraining",
         "Implement rolling walk-forward optimization where models periodically retrain on sliding historical windows (e.g., re-fitting every 252 trading days)."),
        ("5. Volatility Modeling & Live API Feeds",
         "Pair point return estimates with dynamic volatility confidence intervals using GARCH models, and connect directly to automated end-of-day market data APIs."),
    ]

    for idx, (title, desc) in enumerate(scopes):
        top = 1.6 + idx * 1.05
        add_card(slide, 0.8, top, 0.8, 0.9, bg_color=ROYAL_BLUE, border_color=None)
        tb_n = slide.shapes.add_textbox(Inches(0.8), Inches(top + 0.2), Inches(0.8), Inches(0.5))
        p_n = tb_n.text_frame.paragraphs[0]
        p_n.text = f"0{idx+1}"
        p_n.font.name = FONT_TITLE
        p_n.font.size = Pt(16)
        p_n.font.bold = True
        p_n.font.color.rgb = RGBColor(255, 255, 255)
        p_n.alignment = PP_ALIGN.CENTER

        add_card(slide, 1.7, top, 10.8, 0.9, bg_color=CARD_WHITE, border_color=SLATE_BORDER)
        tb_c = slide.shapes.add_textbox(Inches(1.85), Inches(top + 0.08), Inches(10.5), Inches(0.75))
        tf_c = tb_c.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = tf_c.margin_top = 0

        p_t = tf_c.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(13)
        p_t.font.color.rgb = NAVY
        p_t.space_after = Pt(2)

        p_d = tf_c.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = SLATE_DARK

    add_speaker_notes(slide,
        "Looking to the future, there are several promising extensions: "
        "expanding to non-linear models like Random Forests or LSTMs, "
        "integrating alternative data like financial news sentiment through NLP, "
        "incorporating macroeconomic indicators like crude oil and Nifty 50, "
        "and implementing rolling walk-forward cross-validation."
    )


def build_slide_21_conclusion(prs):
    """SLIDE 21: Conclusion & Q&A."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide)
    add_header(slide, "Project Summary & Concluding Remarks", "Conclusion",
               "Key technical takeaways, academic contributions, and final project summary.")
    add_footer(slide, 21)

    # Left: Summary Card
    add_card(slide, 0.8, 1.6, 5.7, 5.15)
    tb1 = slide.shapes.add_textbox(Inches(1.05), Inches(1.8), Inches(5.2), Inches(4.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Key Project Takeaways"
    p1.font.bold = True
    p1.font.size = Pt(16)
    p1.font.color.rgb = NAVY
    p1.space_after = Pt(10)

    points = [
        "End-to-End Working System: Successfully delivered an upload-driven web application in Streamlit and Scikit-Learn with 11 automated workflow stages.",
        "Leakage-Free Methodology: Demonstrated why strict chronological splitting and pipeline-encapsulated scaling are non-negotiable in financial time-series modeling.",
        "Empirical L1 vs L2 Finding: Proved that Lasso outperformed Ridge on unseen data because L1 sparsity zeroed out non-stationary price columns (open, high, low, close), protecting the model from price drift.",
        "Educational & Scientific Integrity: Evaluated models honestly on real test data, debunking naive get-rich-quick claims and emphasizing true signal-to-noise realities.",
        "Robust Quality Assurance: Backed by 31 automated unit and integration tests covering data loading, cleaning, leakage, and prediction.",
    ]
    add_bullet_points(tf1, points, pt_size=11, space_after=8, bold_first_n_words=1)

    # Right: Thank You / Q&A Card
    add_card(slide, 6.8, 1.6, 5.7, 5.15, bg_color=NAVY, border_color=None)
    tb2 = slide.shapes.add_textbox(Inches(7.1), Inches(2.3), Inches(5.1), Inches(3.8))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p_ty = tf2.paragraphs[0]
    p_ty.text = "Thank You!"
    p_ty.font.name = FONT_TITLE
    p_ty.font.size = Pt(32)
    p_ty.font.bold = True
    p_ty.font.color.rgb = RGBColor(255, 255, 255)
    p_ty.space_after = Pt(14)

    p_qa = tf2.add_paragraph()
    p_qa.text = "Questions & Discussion"
    p_qa.font.name = FONT_TITLE
    p_qa.font.size = Pt(18)
    p_qa.font.bold = True
    p_qa.font.color.rgb = TEAL
    p_qa.space_after = Pt(16)

    p_ack = tf2.add_paragraph()
    p_ack.text = "We express our sincere gratitude to our project guide, faculty evaluators, and the department for their continuous guidance and support throughout this project."
    p_ack.font.size = Pt(11.5)
    p_ack.font.color.rgb = RGBColor(203, 213, 225)
    p_ack.space_after = Pt(20)

    p_proj = tf2.add_paragraph()
    p_proj.text = "Project: Stock Return Prediction Using Ridge & Lasso Regression\nApplication: Running locally on Streamlit (localhost:8501)"
    p_proj.font.size = Pt(10)
    p_proj.font.color.rgb = RGBColor(148, 163, 184)

    add_speaker_notes(slide,
        "In conclusion, we have built a complete, working, and leakage-free stock return prediction application. "
        "Our empirical experiment revealed a valuable data science insight: Lasso outperformed Ridge because its L1 penalty "
        "automatically zeroed out non-stationary nominal price levels, allowing the model to generalize much better to future years. "
        "Thank you very much for your time and guidance. We are now happy to answer your questions and demonstrate the live application."
    )


# ===========================================================================
# MAIN GENERATOR FUNCTION
# ===========================================================================
def main():
    print("=" * 60)
    print("Generating PowerPoint Presentation (.pptx)...")
    print(f"Target: {OUTPUT_PPTX_PATH}")
    print("=" * 60)

    prs = create_presentation()

    build_slide_1_title(prs)
    build_slide_2_intro(prs)
    build_slide_3_problem(prs)
    build_slide_4_motivation(prs)
    build_slide_5_objectives(prs)
    build_slide_6_existing(prs)
    build_slide_7_proposed(prs)
    build_slide_8_architecture(prs)
    build_slide_9_methodology(prs)
    build_slide_10_tech_stack(prs)
    build_slide_11_modules(prs)
    build_slide_12_implementation(prs)
    build_slide_13_dataset(prs)
    build_slide_14_algorithm(prs)
    build_slide_15_results(prs)
    build_slide_16_demo(prs)
    build_slide_17_advantages(prs)
    build_slide_18_applications(prs)
    build_slide_19_limitations(prs)
    build_slide_20_future(prs)
    build_slide_21_conclusion(prs)

    prs.save(str(OUTPUT_PPTX_PATH))
    print(f"✅ Successfully created: {OUTPUT_PPTX_PATH.name}")
    print(f"Total Slides: {len(prs.slides)}")
    print(f"File Size: {OUTPUT_PPTX_PATH.stat().st_size:,} bytes")
    print("=" * 60)


if __name__ == "__main__":
    main()
