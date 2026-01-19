import os
import re
import streamlit as st
from openai import OpenAI
import streamlit.components.v1 as components


def scroll_to_bottom():
    components.html(
        """
        <script>
          const scroll = () => window.scrollTo(0, document.body.scrollHeight);
          scroll();
          setTimeout(scroll, 50);
          setTimeout(scroll, 150);
          setTimeout(scroll, 300);
        </script>
        """,
        height=0,
    )


# ---------- STRICT ROUTING HELPERS ----------

MED_INTENT_KEYWORDS = {
    # explicit medication intent
    "medicine", "medication", "drug", "tablet", "pill", "capsule", "syrup",
    "antibiotic", "prescription", "otc", "over the counter",
    "dose", "dosage", "mg", "side effect", "contraindication", "interaction",
    "safe", "safety", "can i take", "should i take", "is it safe", "safe to take",
    "while breastfeeding", "during breastfeeding", "while pregnant", "during pregnancy",
}

DRUG_SUFFIXES = (
    "amide", "olol", "pril", "sartan", "statin", "cillin", "mycin", "cycline",
    "azole", "oxetine", "prazole", "vir", "mab", "nib", "zepam", "zine",
    "triptan", "caine", "thiazide", "gliptin", "gliflozin", "barbital",
)

COMMON_MEDS = {
    "ibuprofen", "acetaminophen", "paracetamol", "tylenol", "advil", "motrin",
    "naproxen", "aleve", "aspirin",
    "amoxicillin", "augmentin", "azithromycin",
    "fluconazole", "diflucan",
    "sertraline", "zoloft",
    "cetirizine", "zyrtec", "loratadine", "claritin", "diphenhydramine", "benadryl",
    "pseudoephedrine",
}

STOPWORDS = {
    "can", "you", "tell", "me", "when", "what", "where", "how", "why", "which",
    "i", "we", "my", "our", "your", "the", "a", "an", "to", "of", "and", "or",
    "stop", "start",
    "breastfeeding", "breast", "feeding", "nursing", "lactation",
    "milk", "supply",
    "baby", "infant", "newborn", "child",
    "bottle", "formula",
    "pump", "pumping",
    "latch", "latching",
    "mastitis", "colostrum", "weaning",
    "during", "while",
    "take", "taking",
    "pregnant", "pregnancy",
    # travel/common query words to prevent accidental detection as drugs
    "airline", "airlines", "flight", "flights", "hotel", "hotels", "travel", "trip",
    "japan", "korea", "india", "usa", "canada",
}


def is_goodbye(question: str) -> bool:
    q = (question or "").strip().lower()
    # match common goodbye forms
    return q in {"bye", "goodbye", "bye!", "goodbye!", "bye.", "goodbye."} or bool(
        re.fullmatch(r"\s*(bye+|goodbye+|see\s+ya|see\s+you|cya|take\s+care)\s*[!.]*\s*", q)
    )


def has_medication_intent(question: str) -> bool:
    q = (question or "").strip().lower()
    return any(k in q for k in MED_INTENT_KEYWORDS)


def extract_drug_candidate(question: str):
    """
    STRICT: Returns a drug candidate only when it looks like an actual medication name.

    Rules:
    - If question has medication intent words, find a token that looks drug-like.
    - If question is just 1-2 tokens, treat as a drug only if it matches suffixes or allowlist.
    """
    q = (question or "").strip().lower()
    tokens = re.findall(r"[a-z][a-z0-9_-]*", q)
    if not tokens:
        return None

    intent = has_medication_intent(q)

    def looks_like_drug(t: str) -> bool:
        if t in STOPWORDS:
            return False
        if t in COMMON_MEDS:
            return True
        if t.endswith(DRUG_SUFFIXES):
            return True
        return False

    if intent:
        for t in tokens:
            if looks_like_drug(t):
                return t
        return None

    if len(tokens) <= 2:
        for t in tokens:
            if looks_like_drug(t):
                return t

    return None


def is_breastfeeding_related(question: str) -> bool:
    q = (question or "").strip().lower()
    bf_keywords = [
        "breastfeeding", "breast feeding", "nursing", "lactation",
        "colostrum", "latch", "latching", "attachment", "position",
        "pump", "pumping", "hand express", "hand expression",
        "engorgement", "mastitis", "nipple", "nipple pain", "sore nipples",
        "cluster feeding", "clusterfeeding",
        "wean", "weaning",
        "milk supply", "low supply", "increase supply", "letdown", "let-down",
        "newborn", "infant", "baby",
        "feeding", "feeding frequency", "hunger cues", "wet diapers",
        "supplementing", "formula", "bottle",
        "tongue tie", "tongue-tie",
    ]
    return any(k in q for k in bf_keywords)


def response_has_file_citations(resp) -> bool:
    try:
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                ann = getattr(c, "annotations", None) or []
                if ann:
                    return True
    except Exception:
        pass
    return False


def append_links(answer: str, medicine_mode: bool, manual_used: bool) -> str:
    answer = (answer or "").rstrip()

    if medicine_mode:
        return (
            answer
            + "\n\n---\n"
            + "Please consult your healthcare provider before taking any medication. "
              "For more details please refer to:\n"
              "https://www.ncbi.nlm.nih.gov/books/NBK501922/\n"
              "\n http://e-lactancia.org/\n"
              "\n https://mothertobaby.org/lactrx/"
            + "\n\nFor Local resources please visit: https://palav.org"
        )

    if not manual_used:
        return answer

    return answer + "\n\n---\n" + "For Local resources please visit: https://palav.org"


# ---------- APP ----------

st.set_page_config(page_title="Breastfeeding Manual Chatbot", layout="centered")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Palav Breastfeeding Userguide")

API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_VS_ID = st.secrets.get("VECTOR_STORE_ID", os.getenv("VECTOR_STORE_ID", ""))

if not API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to Streamlit secrets or your environment variables.")
    st.stop()

client = OpenAI(api_key=API_KEY)

with st.sidebar:
    st.header("Settings")
    vector_store_id = st.text_input("Vector Store ID", value=DEFAULT_VS_ID, placeholder="vs_...")
    model = st.selectbox("Model", ["gpt-4.1-mini"], index=0)
    show_debug = st.checkbox("Show debug", value=False)

if not vector_store_id:
    st.warning("VECTOR_STORE_ID is missing. Add it to Streamlit Secrets as VECTOR_STORE_ID (it starts with `vs_`).")
    st.stop()

MANUAL_MODE_INSTRUCTIONS = (
    "You are an NGO breastfeeding education assistant. "
    "Answer ONLY using the uploaded manual from file_search. "
    "If the manual does not cover the topic, say: "
    "'The manual does not cover this. Please ask different question'. "
    "Use clear, parent-friendly language. "
)

MEDICINE_MODE_INSTRUCTIONS = (
    "You are a breastfeeding and pregnancy medication information assistant. "
    "Use web search to provide a general, high-level summary about the medication and "
    "its pregnancy/breastfeeding considerations. "
    "Do NOT provide dosing instructions. "
    "If evidence is unclear or mixed, say so. "
    "Keep it concise and practical. "
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to Palav Breastfeeding Guide Chatbot.\n\n"
                "I am here to help you find clear, reliable information from Palav breastfeeding user guide.\n\n"
                "You can ask questions or type a topic such as:\n"
                "- early milk or colostrum\n"
                "- Attachment and Latch\n"
                "- Hand Expression Techniques\n"
                "- Feeding frequency and hunger cues\n"
                "- Do and dont for breastfeeding mothers\n"
                "- When to seek medical help\n\n"
                "You can also type any breastfeeding-related topic you would like to learn more about.\n"
                "If you ask about a specific medication name, I will provide a general summary and trusted references."
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Type your question")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # NEW: Goodbye shortcut (no manual checks, no model calls)
    if is_goodbye(prompt):
        answer = "Bye!!! Take care, and feel free to come back anytime you need help or have questions."
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()

    drug_candidate = extract_drug_candidate(prompt)
    medicine_mode = drug_candidate is not None

    # HARD BLOCK: If not medicine mode and not breastfeeding-related -> immediately return not-covered
    if (not medicine_mode) and (not is_breastfeeding_related(prompt)):
        answer = "The manual does not cover this. Please ask different question."
        with st.chat_message("assistant"):
            st.markdown(answer)

        if show_debug:
            with st.expander("Debug"):
                st.write(
                    {
                        "prompt": prompt,
                        "medicine_mode": medicine_mode,
                        "drug_candidate": drug_candidate,
                        "breastfeeding_related": False,
                        "blocked_off_topic": True,
                    }
                )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Searching userguide..."):
            try:
                if medicine_mode:
                    resp = client.responses.create(
                        model=model,
                        instructions=MEDICINE_MODE_INSTRUCTIONS,
                        input=[{"role": "user", "content": prompt}],
                        tools=[{"type": "web_search", "search_context_size": "low"}],
                        timeout=90,
                    )
                    raw_answer = getattr(resp, "output_text", "").strip() or "(No output received.)"
                    answer = append_links(raw_answer, medicine_mode=True, manual_used=False)
                else:
                    input_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    resp = client.responses.create(
                        model=model,
                        instructions=MANUAL_MODE_INSTRUCTIONS,
                        input=input_messages,
                        tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
                        timeout=90,
                    )

                    raw_answer = getattr(resp, "output_text", "").strip() or "(No output received.)"
                    used_manual = response_has_file_citations(resp)

                    if not used_manual:
                        raw_answer = "The manual does not cover this. Please ask different question."

                    answer = append_links(raw_answer, medicine_mode=False, manual_used=used_manual)

                st.markdown(answer)

                if show_debug:
                    with st.expander("Debug"):
                        st.write(
                            {
                                "prompt": prompt,
                                "medicine_mode": medicine_mode,
                                "drug_candidate": drug_candidate,
                                "breastfeeding_related": is_breastfeeding_related(prompt),
                                "manual_used_citations": (response_has_file_citations(resp) if not medicine_mode else None),
                            }
                        )

            except Exception as e:
                st.exception(e)
                answer = "Sorry - something went wrong while answering. Please try again."
                st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

scroll_to_bottom()
