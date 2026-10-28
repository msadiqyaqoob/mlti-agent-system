
from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain
)


def get_last_message_content(result):
    """Agent response se last message ka content nikalo."""

    messages = result.get("messages", [])

    if not messages:
        raise ValueError(
            f"Agent ne koi messages return nahi kiye: {result}"
        )

    return messages[-1].content


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # STEP 1: Search Agent
    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"Search for recent and reliable information "
                f"on the topic: {topic}. "
                "Return useful findings with their source URLs."
            )
        ]
    })

    state["search_result"] = get_last_message_content(
        search_result
    )

    print("\n--- SEARCH RESULTS ---")
    print(state["search_result"][:1000])

    # STEP 2: Reader Agent
    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"Based on these search results about '{topic}', "
                "extract a valid HTTP or HTTPS URL and scrape "
                "the most relevant source. If no valid URL exists, "
                "explain that no URL was found.\n\n"
                f"Search Results:\n{state['search_result'][:3000]}"
            )
        ]
    })

    state["scraped_content"] = get_last_message_content(
        reader_result
    )

    print("\n--- SCRAPED CONTENT ---")
    print(state["scraped_content"][:1000])

    # STEP 3: Combine research
    research_combined = (
        f"Search Results:\n{state['search_result']}\n\n"
        f"Scraped Content:\n{state['scraped_content']}"
    )

    # STEP 4: Writer Chain
    writer_result = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    state["research_report"] = writer_result

    # STEP 5: Critic Chain
    critic_result = critic_chain.invoke({
        "report": state["research_report"]
    })

    state["critic_feedback"] = critic_result

    print("\n--- RESEARCH REPORT ---")
    print(state["research_report"])

    print("\n--- CRITIC FEEDBACK ---")
    print(state["critic_feedback"])

    return state


if __name__ == "__main__":
    topic = input("Enter the research topic: ")
    run_research_pipeline(topic)
