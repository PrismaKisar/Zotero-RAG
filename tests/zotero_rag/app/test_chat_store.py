"""Chats and runs in one SQLite file, on a temporary path."""



from zotero_rag.app.chat_store import ChatStore


def test_chats_come_back_last_changed_first_and_can_be_deleted(tmp_path):
    store = ChatStore(tmp_path / "nested" / "app.db")
    store.save_chat("a", "First", [{"kind": "ask"}])
    store.save_chat("b", "Second", [])
    store.save_chat("a", "First, renamed", [{"kind": "ask"}, {"kind": "index"}])

    assert [(c["id"], c["title"], len(c["items"])) for c in store.chats()] == [
        ("a", "First, renamed", 2), ("b", "Second", 0)]
    store.delete_chat("a")
    assert [c["id"] for c in store.chats()] == ["b"]


def test_a_new_store_on_the_same_file_sees_what_the_old_one_wrote(tmp_path):
    ChatStore(tmp_path / "app.db").save_run("r", [{"text": "42"}])
    assert ChatStore(tmp_path / "app.db").run("r") == [{"text": "42"}]
    assert ChatStore(tmp_path / "app.db").run("missing") is None
