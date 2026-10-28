import streamlit as st

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)
from pipeline import get_last_message_content

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)


def to_text(content) -> str:
    """Some models return a list of content blocks instead of a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def run_pipeline_with_progress(topic: str) -> dict:
    """Same steps as pipeline.run_research_pipeline, but with live UI status."""
    state = {}

    with st.status("Running research pipeline...", expanded=True) as status:

        # STEP 1: Search Agent
        st.write("🔎 Step 1/4: Searching the web...")
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [(
                "user",
                f"Search for recent and reliable information "
                f"on the topic: {topic}. "
                "Return useful findings with their source URLs.",
            )]
        })
        state["search_result"] = to_text(get_last_message_content(search_result))

        # STEP 2: Reader Agent
        st.write("📖 Step 2/4: Reading the most relevant source...")
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [(
                "user",
                f"Based on these search results about '{topic}', "
                "extract a valid HTTP or HTTPS URL and scrape "
                "the most relevant source. If no valid URL exists, "
                "explain that no URL was found.\n\n"
                f"Search Results:\n{state['search_result'][:3000]}",
            )]
        })
        state["scraped_content"] = to_text(get_last_message_content(reader_result))

        # STEP 3: Writer
        st.write("✍️ Step 3/4: Writing the report...")
        research_combined = (
            f"Search Results:\n{state['search_result']}\n\n"
            f"Scraped Content:\n{state['scraped_content']}"
        )
        writer_result = writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
        state["research_report"] = to_text(writer_result)

        # STEP 4: Critic
        st.write("🧐 Step 4/4: Reviewing the report...")
        critic_result = critic_chain.invoke({
            "report": state["research_report"],
        })
        state["critic_feedback"] = to_text(critic_result)

        status.update(label="Research complete!", state="complete", expanded=False)

    return state


# ---------------- UI ----------------

st.title("🔎 Multi-Agent Research System")
st.caption("Search Agent → Reader Agent → Writer → Critic")

with st.sidebar:
    st.header("About")
    st.markdown(
        "1. **Search Agent** finds recent sources\n"
        "2. **Reader Agent** scrapes the best one\n"
        "3. **Writer** drafts the report\n"
        "4. **Critic** reviews it"
    )
    if st.button("Clear results"):
        st.session_state.pop("result", None)
        st.session_state.pop("topic", None)
        st.rerun()

topic = st.text_input(
    "Research topic",
    placeholder="e.g. Latest advances in quantum computing",
)

run = st.button("Start Research", type="primary", disabled=not topic.strip())

if run:
    try:
        st.session_state["result"] = run_pipeline_with_progress(topic.strip())
        st.session_state["topic"] = topic.strip()
    except Exception as e:
        st.error(f"Pipeline failed: {e}")

result = st.session_state.get("result")

if result:
    st.subheader(f"Results: {st.session_state.get('topic', '')}")

    tab_report, tab_critic, tab_search, tab_scraped = st.tabs(
        ["📄 Report", "🧐 Critic Feedback", "🔎 Search Results", "📖 Scraped Content"]
    )

    with tab_report:
        st.markdown(result["research_report"])
        st.download_button(
            "Download report (.md)",
            data=result["research_report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_critic:
        st.markdown(result["critic_feedback"])

    with tab_search:
        st.markdown(result["search_result"])

    with tab_scraped:
        st.markdown(result["scraped_content"])