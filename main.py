from agent.graph import run_graph, answer_question


def main():
    query = input("Enter an arXiv paper ID, URL, or research topic: ").strip()

    if not query:
        print("Please enter a query.")
        return

    state = run_graph(query)

    if state.status != "briefing_generated":
        print(f"\nAgent failed with status: {state.status}")

        if state.errors:
            print("\nErrors:")
            for error in state.errors:
                print(f"- {error}")

        return

    print("\n" + "=" * 80)
    print("EXECUTIVE BRIEFING")
    print("=" * 80)
    print(state.briefing["content"])

    print("\n" + "=" * 80)
    print("PAPER QA")
    print("=" * 80)
    print("Ask questions about the paper.")
    print("Press Enter without a question to exit.")

    while True:
        question = input("\nQuestion: ").strip()

        if not question:
            break

        state = answer_question(state, question)

        if state.status != "qa_answered":
            print("\nUnable to answer the question.")

            if state.errors:
                print("Errors:")
                for error in state.errors:
                    print(f"- {error}")

            break

        print("\nAnswer:")
        print(state.conversation_history[-1]["content"])


if __name__ == "__main__":
    main()