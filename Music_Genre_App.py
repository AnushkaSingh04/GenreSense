import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from PIL import Image

# ============================================================
# GENRESENSE — Music Genre Classification
# ============================================================

st.set_page_config(
    page_title="GenreSense | Music Intelligence",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
HERO_IMAGE = BASE_DIR / "music_genre_home.png"

# The project contains both legacy H5 and newer Keras model files.
# Prefer the H5 file because the original project was written to load
# Trained_model.h5, and it is usually the safest format for the
# TensorFlow/Keras version used by this project.
MODEL_CANDIDATES = [
    BASE_DIR / "Trained_model.h5",
    BASE_DIR / "Trained_model.keras",
]

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

GENRE_EMOJI = {
    "blues": "🎷",
    "classical": "🎻",
    "country": "🤠",
    "disco": "🪩",
    "hiphop": "🎤",
    "jazz": "🎺",
    "metal": "🤘",
    "pop": "🎧",
    "reggae": "🌴",
    "rock": "🎸",
}


# ----------------------------- CSS -----------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #090817;
        --panel: #12102a;
        --panel2: #171333;
        --text: #f8f7ff;
        --muted: #a9a5c3;
        --pink: #ff3cac;
        --purple: #7b2cff;
        --cyan: #21d4fd;
        --border: rgba(255,255,255,.10);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(123,44,255,.18), transparent 30%),
            radial-gradient(circle at 90% 10%, rgba(255,60,172,.12), transparent 28%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: rgba(10, 8, 28, .96);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: #eeeaff;
    }

    .brand {
        padding: 8px 8px 26px;
        text-align: center;
    }

    .brand-icon {
        width: 58px;
        height: 58px;
        margin: 0 auto 12px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 27px;
        background: linear-gradient(135deg, var(--purple), var(--pink));
        box-shadow: 0 0 35px rgba(255,60,172,.25);
    }

    .brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -.7px;
    }

    .brand-sub {
        color: var(--muted);
        font-size: 12px;
        margin-top: 3px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 48px 52px;
        border: 1px solid var(--border);
        border-radius: 30px;
        background:
            linear-gradient(110deg, rgba(10,8,28,.98) 15%, rgba(27,15,57,.88) 65%, rgba(79,25,91,.55)),
            radial-gradient(circle at 85% 50%, rgba(255,60,172,.22), transparent 34%);
        box-shadow: 0 25px 80px rgba(0,0,0,.30);
    }

    .hero-kicker {
        color: #ff75c8;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 15px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(42px, 5vw, 72px);
        line-height: .98;
        letter-spacing: -3px;
        max-width: 720px;
        margin: 0;
    }

    .gradient-text {
        background: linear-gradient(90deg, #ff55ba, #a96bff, #47dfff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        color: #c2bdd8;
        font-size: 16px;
        line-height: 1.7;
        max-width: 650px;
        margin: 22px 0 0;
    }

    .hero-image {
        width: 100%;
        border-radius: 22px;
        margin-top: 32px;
        border: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 20px 55px rgba(0,0,0,.32);
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 30px;
        margin: 42px 0 6px;
    }

    .section-sub {
        color: var(--muted);
        margin-bottom: 24px;
    }

    .stat-card {
        background: linear-gradient(145deg, rgba(25,20,55,.94), rgba(15,13,35,.94));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 23px;
        min-height: 112px;
    }

    .stat-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 30px;
        font-weight: 700;
    }

    .stat-label {
        color: var(--muted);
        font-size: 13px;
        margin-top: 5px;
    }

    .genre-pill {
        display: inline-block;
        padding: 8px 13px;
        margin: 4px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.10);
        background: rgba(255,255,255,.045);
        color: #e9e5ff;
        font-size: 13px;
    }

    .glass-card {
        background: rgba(20,17,45,.78);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 28px;
        box-shadow: 0 20px 60px rgba(0,0,0,.22);
    }

    .upload-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        margin-bottom: 6px;
    }

    .upload-sub {
        color: var(--muted);
        margin-bottom: 18px;
    }

    .result-card {
        padding: 30px;
        border-radius: 26px;
        border: 1px solid rgba(255,60,172,.30);
        background:
            radial-gradient(circle at 90% 0%, rgba(255,60,172,.18), transparent 35%),
            linear-gradient(145deg, rgba(36,21,70,.95), rgba(17,13,40,.95));
        box-shadow: 0 20px 70px rgba(123,44,255,.15);
    }

    .result-label {
        color: #aaa4c5;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 700;
    }

    .result-genre {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 48px;
        font-weight: 700;
        margin: 4px 0 10px;
        text-transform: capitalize;
    }

    .confidence {
        color: #c7c1dc;
        font-size: 14px;
    }

    .about-card {
        background: rgba(20,17,45,.72);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 26px;
        margin-bottom: 18px;
    }

    .about-card h3 {
        font-family: 'Space Grotesk', sans-serif;
        margin-top: 0;
    }

    .muted {
        color: var(--muted);
        line-height: 1.7;
    }

    .footer {
        text-align: center;
        color: #77718f;
        padding: 50px 0 20px;
        font-size: 12px;
    }

    div.stButton > button {
        width: 100%;
        border: 0;
        border-radius: 13px;
        padding: 13px 18px;
        font-weight: 700;
        color: white;
        background: linear-gradient(90deg, #7b2cff, #ff3cac);
        box-shadow: 0 10px 28px rgba(255,60,172,.18);
    }

    div.stButton > button:hover {
        border: 0;
        color: white;
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.025);
        border: 1px dashed rgba(255,255,255,.20);
        border-radius: 18px;
        padding: 8px;
    }

    .stAlert {
        border-radius: 14px;
    }

    @media (max-width: 700px) {
        .hero { padding: 30px 24px; }
        .hero-title { letter-spacing: -2px; }
        .result-genre { font-size: 38px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------- Model ------------------------------

@st.cache_resource
def load_model():
    errors = []

    for model_path in MODEL_CANDIDATES:
        if not model_path.exists():
            continue

        try:
            # compile=False avoids loading optimizer/training state during
            # inference and is more robust across TensorFlow/Keras versions.
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        except Exception as e:
            errors.append(f"{model_path.name}: {type(e).__name__}: {e}")

    if not errors:
        names = ", ".join(p.name for p in MODEL_CANDIDATES)
        raise FileNotFoundError(
            f"No trained model was found. Expected one of: {names}"
        )

    raise RuntimeError(
        "The model files were found, but neither could be loaded. "
        "This usually means the model was saved with a different "
        "TensorFlow/Keras version.\n\n" + "\n\n".join(errors)
    )


def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    chunk_duration = 4
    overlap_duration = 2

    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    step = max(1, chunk_samples - overlap_samples)

    if len(audio_data) <= chunk_samples:
        starts = [0]
    else:
        starts = range(0, len(audio_data) - chunk_samples + 1, step)

    data = []

    for start in starts:
        chunk = audio_data[start:start + chunk_samples]

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        mel = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel = tf.image.resize(
            np.expand_dims(mel, axis=-1),
            target_shape
        ).numpy()

        data.append(mel)

    return np.asarray(data)


def predict_genre(X_test):
    model = load_model()
    predictions = model.predict(X_test, verbose=0)

    # Average probabilities across audio chunks.
    mean_probabilities = np.mean(predictions, axis=0)
    result_index = int(np.argmax(mean_probabilities))

    return (
        GENRES[result_index],
        float(mean_probabilities[result_index]),
        mean_probabilities
    )


def make_spectrogram(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor("#100d25")
    ax.set_facecolor("#100d25")

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma"
    )

    ax.set_title("Mel Spectrogram", color="white", fontsize=12, pad=10)
    ax.tick_params(colors="#aaa4c5")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=.01)
    fig.tight_layout()

    return fig


# ------------------------- Sidebar -----------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🎵</div>
            <div class="brand-name">GenreSense</div>
            <div class="brand-sub">Music Intelligence • CNN</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "EXPLORE",
        ["⌂  Home", "◉  Classify", "◎  About"],
        label_visibility="visible",
    )

    st.divider()

    st.markdown(
        """
        <div class="muted">
        <b>Powered by</b><br>
        TensorFlow · Librosa · Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------- HOME ------------------------------

if page == "⌂  Home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">AI • AUDIO • DEEP LEARNING</div>
            <h1 class="hero-title">
                Give your music<br>
                a <span class="gradient-text">genre.</span>
            </h1>
            <p class="hero-desc">
                GenreSense analyzes an audio track through Mel Spectrograms
                and a Convolutional Neural Network to predict its musical genre.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if HERO_IMAGE.exists():
        try:
            hero_img = Image.open(HERO_IMAGE)
            st.image(hero_img, use_column_width=True)
        except Exception:
            st.error("Could not load music_genre_home.png. Please check that the PNG file is valid.")
    else:
        st.warning("music_genre_home.png was not found in the app folder.")

    st.markdown('<div class="section-title">What happens under the hood?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">From raw audio to an intelligent genre prediction.</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(4)
    steps = [
        ("01", "Upload", "Add an MP3, WAV, OGG or FLAC audio file."),
        ("02", "Transform", "Convert the audio into Mel Spectrogram features."),
        ("03", "Analyze", "The CNN evaluates learned visual audio patterns."),
        ("04", "Predict", "The strongest genre probability becomes the result."),
    ]

    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="result-label">{num}</div>
                    <div style="font-family:Space Grotesk;font-size:19px;font-weight:700;margin-top:7px">{title}</div>
                    <div class="stat-label">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">10 genres. One model.</div>', unsafe_allow_html=True)
    pills = "".join(
        f'<span class="genre-pill">{GENRE_EMOJI[g]} {g.capitalize()}</span>'
        for g in GENRES
    )
    st.markdown(pills, unsafe_allow_html=True)

    st.markdown('<div class="footer">GenreSense • Music Genre Classification with Convolutional Neural Network</div>', unsafe_allow_html=True)


# ------------------------- CLASSIFY ----------------------------

elif page == "◉  Classify":

    st.markdown(
        """
        <div class="hero" style="padding:38px 42px">
            <div class="hero-kicker">GENRE CLASSIFIER</div>
            <h1 class="hero-title" style="font-size:48px">
                What's the <span class="gradient-text">vibe?</span>
            </h1>
            <p class="hero-desc">
                Upload a track and let the model listen to its audio features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    left, right = st.columns([1.05, .95], gap="large")

    with left:
        st.markdown(
            """
            <div class="glass-card">
                <div class="upload-title">🎧 Upload your track</div>
                <div class="upload-sub">
                    Supported formats: MP3, WAV, OGG, FLAC
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Drop your audio here",
            type=["mp3", "wav", "ogg", "flac"],
            label_visibility="collapsed",
        )

        if uploaded:
            suffix = Path(uploaded.name).suffix or ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                temp_path = tmp.name

            st.audio(uploaded)

            try:
                duration = librosa.get_duration(path=temp_path)
                audio, sr = librosa.load(temp_path, sr=None, mono=True)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Duration", f"{duration:.1f}s")
                with m2:
                    st.metric("Sample Rate", f"{sr:,} Hz")
                with m3:
                    st.metric("Samples", f"{len(audio):,}")

                with st.expander("View audio representation"):
                    fig = make_spectrogram(temp_path)
                    st.pyplot(fig)
                    plt.close(fig)

            except Exception as e:
                st.warning(f"Could not read audio metadata: {e}")

    with right:
        st.markdown(
            """
            <div class="glass-card">
                <div class="upload-title">🧠 AI Analysis</div>
                <div class="upload-sub">
                    The model converts your audio into Mel Spectrogram chunks
                    and evaluates them using the trained CNN.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if uploaded:
            if st.button("✨ Analyze Genre", use_container_width=True):
                try:
                    with st.spinner("Listening to the audio..."):
                        X_test = load_and_preprocess_data(temp_path)
                        genre, confidence, probabilities = predict_genre(X_test)

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">PREDICTED GENRE</div>
                            <div class="result-genre">
                                {GENRE_EMOJI.get(genre, "🎵")} {genre}
                            </div>
                            <div class="confidence">
                                Model confidence: <b>{confidence * 100:.1f}%</b>
                                &nbsp; • &nbsp; {len(X_test)} audio chunk(s) analyzed
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.write("")
                    st.markdown("#### Probability distribution")

                    order = np.argsort(probabilities)[::-1]
                    for idx in order[:5]:
                        st.progress(
                            float(probabilities[idx]),
                            text=f"{GENRE_EMOJI[GENRES[idx]]} {GENRES[idx].capitalize()} — {probabilities[idx] * 100:.1f}%"
                        )

                except Exception as e:
                    st.error(
                        "Prediction could not be completed. "
                        "See the diagnostic below; the app will identify "
                        "whether the model file or TensorFlow/Keras version "
                        "is the issue."
                    )
                    st.exception(e)
        else:
            st.info("Upload an audio file to activate the classifier.")

    st.markdown('<div class="footer">Your audio is processed locally by the Streamlit application.</div>', unsafe_allow_html=True)


# --------------------------- ABOUT -----------------------------

else:

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">ABOUT THE PROJECT</div>
            <h1 class="hero-title">
                Turning <span class="gradient-text">sound</span>
                into insight.
            </h1>
            <p class="hero-desc">
                A deep-learning based music genre classification system built
                around audio feature extraction and Convolutional Neural Networks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">How GenreSense works</div>', unsafe_allow_html=True)

    about_cards = [
        (
            "🎼 Audio Representation",
            "Raw audio is converted into Mel Spectrograms, creating a visual representation of the frequency content of the music."
        ),
        (
            "🧠 Convolutional Neural Network",
            "The trained CNN learns patterns from these spectrogram representations and maps them to music genres."
        ),
        (
            "📊 Multi-class Prediction",
            "The system predicts among 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock."
        ),
        (
            "⚡ Chunk-based Analysis",
            "Longer tracks are divided into overlapping audio chunks. Predictions are aggregated to produce the final genre."
        ),
    ]

    for title, text in about_cards:
        st.markdown(
            f"""
            <div class="about-card">
                <h3>{title}</h3>
                <div class="muted">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Technology stack</div>', unsafe_allow_html=True)

    techs = ["Python", "TensorFlow", "Keras", "Librosa", "NumPy", "Matplotlib", "Streamlit"]
    st.markdown(
        "".join(f'<span class="genre-pill">{x}</span>' for x in techs),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="about-card">
            <div class="muted">
            The project documentation describes a dataset containing 10 genres
            with 100 audio files per genre, each 30 seconds long. The audio is
            represented using Mel Spectrograms so that the CNN can learn
            discriminative visual patterns.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="footer">Built with Python + Deep Learning + a little love for music 🎵</div>', unsafe_allow_html=True)
