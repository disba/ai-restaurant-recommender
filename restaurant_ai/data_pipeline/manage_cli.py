"""
Module 1 / Step 4 - Command-line manager for the restaurant database.

Browse, view, add (LLM-structured), edit and delete restaurant records.
Every change backs up the JSON file first and updates the vector index so
the agents immediately "see" the change.

Run:
    python -m restaurant_ai.data_pipeline.manage_cli
"""

from restaurant_ai.data_pipeline import store

MENU = """
=== RESTAURANT DATABASE ({count} records) ===
1. Browse all (names)
2. View a restaurant
3. Add new restaurant (from a free-text description, structured by the LLM)
4. Edit a restaurant
5. Delete a restaurant
6. Exit
"""
EDITABLE_TEXT_FIELDS = ["name", "location", "type", "food_style", "vibe", "environment"]


def ask_item_id() -> int | None:
    try:
        return int(input("Enter itemId (e.g. 1000001): "))
    except ValueError:
        print("Invalid itemId.")
        return None


def confirm_write() -> bool:
    answer = input("This will change the database (a backup is made first). Type 'yes' to continue: ")
    return answer.strip().lower() == "yes"


def sync_index(record: dict | None = None, deleted_id: int | None = None):
    """Keep the vector DB in sync. Skipped gracefully if the index hasn't been built yet."""
    try:
        from restaurant_ai.rag import retriever
        if record:
            retriever.upsert_restaurant(record)
        if deleted_id:
            retriever.delete_restaurant(deleted_id)
    except Exception as error:
        print(f"(vector index not updated: {error})")


def main():
    while True:
        restaurants = store.load_restaurants()
        print(MENU.format(count=len(restaurants)))
        choice = input("Action: ").strip()

        if choice == "1":
            for r in restaurants:
                print(f"{r.get('itemId')}  {r.get('name') or 'N/A'}  ({r.get('location') or '?'})")

        elif choice == "2":
            item_id = ask_item_id()
            match = next((r for r in restaurants if r.get("itemId") == item_id), None)
            print(store.format_restaurant_card(match) if match else "Not found.")

        elif choice == "3" and confirm_write():
            paragraph = input("Paste the restaurant description:\n> ")
            record = store.add_restaurant_from_text(paragraph)
            sync_index(record=record)
            print("Added:\n" + store.format_restaurant_card(record))

        elif choice == "4" and confirm_write():
            item_id = ask_item_id()
            changes = {}
            for field in EDITABLE_TEXT_FIELDS:
                value = input(f"New {field} (Enter to keep): ").strip()
                if value:
                    changes[field] = value
            try:
                record = store.update_restaurant(item_id, changes)
                sync_index(record=record)
                print("Updated:\n" + store.format_restaurant_card(record))
            except KeyError as error:
                print(error)

        elif choice == "5" and confirm_write():
            item_id = ask_item_id()
            try:
                removed = store.delete_restaurant(item_id)
                sync_index(deleted_id=item_id)
                print(f"Deleted: {removed.get('name')}")
            except KeyError as error:
                print(error)

        elif choice == "6":
            break


if __name__ == "__main__":
    main()
