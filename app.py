"""Reporter Hasan TTS — KIKI Retrain (2550) (Streamlit Community Cloud).

লেখা লিখুন -> আপনার কণ্ঠে (KIKI Retrain — ২,৫৫০ ক্লিপ) অডিও বানান -> শুনুন / WAV ডাউনলোড করুন।
মডেল ফাইল GitHub-এ রাখা হয় না (১১৪ MB, সীমার বেশি) — অ্যাপ প্রথমবার
চালু হলে Hugging Face থেকে নামিয়ে MD5 মিলিয়ে নেয়, তারপর cache করে।
"""

import time

import streamlit as st

from tts_engine import (
    DEFAULT_TEXT,
    SAMPLE_GENERAL,
    SAMPLE_NEWS,
    SAMPLE_RATE,
    ensure_model,
    samples_to_wav_bytes,
    synthesize,
)

st.set_page_config(page_title="Reporter Hasan TTS — KIKI Retrain (2550)", layout="centered")
st.title("Reporter Hasan TTS — KIKI Retrain (2550)")
st.caption("বাংলা লেখা লিখুন, আপনার কণ্ঠে (KIKI Retrain — ২,৫৫০ ক্লিপ) অডিও বানান।")

# --- নমুনা লেখা ---
col1, col2 = st.columns(2)
with col1:
    if st.button("📰 নমুনা খবর", use_container_width=True):
        st.session_state["text"] = SAMPLE_NEWS
with col2:
    if st.button("💬 সাধারণ লেখা", use_container_width=True):
        st.session_state["text"] = SAMPLE_GENERAL

text = st.text_area(
    "লেখা",
    value=st.session_state.get("text", DEFAULT_TEXT),
    height=140,
    key="text",
    help="দাঁড়ি (।) দিয়ে বাক্য শেষ করুন — লম্বা লেখা দাঁড়ি ধরে ভাগ করে বানানো হয়।",
)

if st.button("🔊 অডিও বানাও", type="primary", use_container_width=True):
    text = (text or "").strip()
    if not text:
        st.warning("আগে কিছু লিখুন।")
    else:
        with st.spinner("মডেল লোড হচ্ছে / অডিও বানানো হচ্ছে… (প্রথমবার মডেল নামতে একটু সময় লাগবে)"):
            try:
                t0 = time.time()
                ensure_model()
                samples, info = synthesize(text)
                total = time.time() - t0
            except Exception as exc:  # noqa: BLE001 -- show the real error to the user
                st.error(f"অডিও বানানো যায়নি: {exc}")
            else:
                wav = samples_to_wav_bytes(samples)
                st.audio(wav, format="audio/wav", sample_rate=SAMPLE_RATE)
                st.download_button(
                    "⬇️ WAV ডাউনলোড করুন",
                    data=wav,
                    file_name="reporter-hasan-tts.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )
                st.success(
                    f"অডিও {info['duration_s']} সেকেন্ড, "
                    f"বানাতে লেগেছে {total:.0f} সেকেন্ড "
                    f"({info['parts']} ভাগে ভাগ করে)"
                )
                st.caption(
                    f"peak {info['peak']} · RMS {info['rms']} · "
                    f"RTF {info['rtf']} (১-এর নিচে মানে অডিওর চেয়ে দ্রুত)"
                )

st.divider()
st.caption(
    "ফ্রি tier CPU-মাত্র, তাই ১ মিনিটের অডিও বানাতে ~১.৫–২.৫ মিনিট লাগতে পারে। "
    "১২ ঘণ্টা কেউ না এলে অ্যাপ ঘুমিয়ে যায় — প্রথমবার খুললে ~১ মিনিট অপেক্ষা করতে হবে। "
    "মডেল: KIKI Retrain (2550) — ২,৫৫০টি ক্লিপ, ৯৭ep থেকে continue, ১২ epoch — Hugging Face থেকে লোড হয়।"
)
