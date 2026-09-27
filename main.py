from agent.graph import run_graph


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


if __name__ == "__main__":
    main()
