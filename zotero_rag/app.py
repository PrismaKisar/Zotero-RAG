"""
app.py - Streamlit web interface for Zotero RAG System

Run with: streamlit run app.py
"""

import os
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict

import streamlit as st
from embedding_manager import DEFAULT_ENCODE_BATCH_SIZE
from highlighter import highlight_pdf
from pdf_utils import sanitize_filename
from pipeline import ZoteroRAG
from reranker import DEFAULT_RERANK_BATCH_SIZE
from zotero_db import ZoteroDatabase


def load_zotero_collections():
    try:
        with st.spinner("Loading Zotero collections..."):
            st.session_state.zotero_collections = ZoteroDatabase(None).list_collections()
            st.session_state.zotero_collections_loaded = True
    except sqlite3.OperationalError as e:
        if "locked" in str(e):
            st.error("Zotero database is locked")
            st.warning("""
            **The database is currently locked by Zotero.**

            You have two options:

            1. **Close Zotero** (Recommended)
            - Close the Zotero application completely
            - Then refresh this page

            2. **Keep Zotero open** (Advanced)
            - The app will try to read the database in read-only mode
            - Click the button below to retry
            """)

            if st.button("Retry Connection"):
                st.rerun()
        else:
            st.error(f"Database error: {e}")
        st.stop()
    except Exception as e:  # noqa: BLE001 - UI boundary: report, never crash the app
        st.error(f"Error loading Zotero: {e}")
        st.info("Make sure Zotero is installed and the database is accessible")


def show_renamed_titles(items, summary, label):
    """List ``{input_title, indexed_title}`` entries, showing a rename as ``a -> b``."""
    if not items:
        return
    st.info(f"{len(items)} {summary}")
    with st.expander(label):
        for item in items:
            input_title = item.get("input_title", "")
            indexed_title = item.get("indexed_title", "")
            if indexed_title and indexed_title != input_title:
                st.write(f"- {input_title} -> {indexed_title}")
            else:
                st.write(f"- {input_title or indexed_title}")


def run_ingest_and_index(result):
    if result is None:
        st.warning("No ingest result returned.")
        return

    if result.failed_pdfs:
        st.warning(f"{len(result.failed_pdfs)} PDF could not be uploaded to cache.")
        with st.expander("Show upload issues"):
            for item in result.failed_pdfs:
                st.write(f"- {item.get('title', 'Unknown file')}: {item.get('error', 'Unknown error')}")

    ingested_pdfs = result.ingested_pdfs

    if not ingested_pdfs:
        st.info("No new PDF to index.")
        return

    newly_cached_count = sum(item.newly_cached for item in ingested_pdfs)
    if newly_cached_count < len(ingested_pdfs):
        st.info(
            f"Using cache for {len(ingested_pdfs) - newly_cached_count} PDF(s) already stored by hash."
        )

    stage_status = st.empty()
    pdf_progress_bar = st.progress(0, text="PDF analysis pending...")
    contextualization_progress_bar = st.progress(0, text="Contextualization pending...")
    encoding_progress_bar = st.progress(0, text="Embedding pending...")
    upsert_progress_bar = st.progress(0, text="Qdrant upsert pending...")

    def progress_callback(stage, current, total, message):
        progress = current / total if total > 0 else 0
        stage_status.info(message)
        if stage == 'pdf':
            pdf_progress_bar.progress(progress, text=f"PDF analysis: {message}")
        elif stage == 'contextualization':
            contextualization_progress_bar.progress(progress, text=f"Contextualization: {message}")
        elif stage == 'encoding':
            encoding_progress_bar.progress(progress, text=f"Embedding: {message}")
        elif stage == 'upserting':
            upsert_progress_bar.progress(progress, text=f"Qdrant upsert: {message}")

    try:
        upsert_report = st.session_state.rag.upsert_pdfs(
            target_pdfs=ingested_pdfs,
            progress_callback=progress_callback
        )

        indexed_chunks = upsert_report.indexed_chunks
        processed_pdfs = upsert_report.processed_pdfs
        failed_pdfs = upsert_report.failed_pdfs
        duplicate_titles = upsert_report.duplicate_titles

        show_renamed_titles(upsert_report.already_indexed,
                            "PDF(s) skipped because already indexed in Qdrant.",
                            "Show Qdrant duplicates")

        if duplicate_titles:
            st.warning(
                f"{len(duplicate_titles)} PDF(s) skipped because the title is already in use."
            )
            with st.expander("Show duplicate titles"):
                for title in duplicate_titles:
                    st.write(f"- {title}")

        show_renamed_titles(upsert_report.title_overrides,
                            "PDF(s) use existing registry titles.",
                            "Show registry title overrides")

        if failed_pdfs:
            st.warning(f"{len(failed_pdfs)} PDF failed during processing.")
            grobid_errors = [
                item for item in failed_pdfs
                if "GROBID service not reachable" in (item.get("error") or "")
            ]
            if grobid_errors:
                st.error("GROBID is not running. Start the service and retry indexing.")
            with st.expander("Show processing errors"):
                for item in failed_pdfs:
                    st.write(f"- {item.get('title', 'Unknown PDF')}: {item.get('error', 'Unknown error')}")

        stage_status.success("Indexing workflow completed.")
        pdf_progress_bar.progress(1.0, text="PDF analysis completed")
        contextualization_progress_bar.progress(1.0, text="Contextualization completed")
        encoding_progress_bar.progress(1.0, text="Embedding completed")
        upsert_progress_bar.progress(1.0, text="Qdrant upsert completed")

        if indexed_chunks > 0:
            st.session_state.indexed = True
            st.success(
                f"Indexing complete: {processed_pdfs} new PDFs processed, {indexed_chunks} chunks upserted."
            )
        else:
            st.info("Indexing finished but no new chunks were upserted.")

        time.sleep(5.0)
        st.rerun()
    except ConnectionError as e:
        if "Qdrant" in str(e):
            st.error("Qdrant is not running. Start the service and retry indexing.")
        elif "GROBID" in str(e):
            st.error("GROBID is not running. Start the service and retry indexing.")
        else:
            st.error(f"Connection error: {e}")
    except Exception as e:  # noqa: BLE001 - UI boundary: report, never crash the app
        stage_status.error("Indexing failed.")
        with st.expander("Show full error"):
            st.exception(e)

def main():
    st.set_page_config(
        page_title="Zotero RAG Navigator",
        page_icon="",
        layout="wide"
    )

    st.title("Zotero RAG Navigator")
    st.markdown("Query your local Zotero library with natural language")

    # Output directory (first thing asked)
    st.subheader("Output Directory")
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = os.environ.get("OUTPUT_DIR", "./output")
    output_dir = st.text_input(
        "Base output directory (TEI cache, highlights)",
        value=st.session_state.output_dir,
        help="Choose where to store cached TEI files, and highlighted PDFs."
    )
    st.session_state.output_dir = output_dir or "./output"

    for key, default in {
        'rag': None,
        'search_results': [],
        'current_answer': 0,
        'indexed': False,
        'zotero_collections_loaded': False,
        'zotero_collections': [],
        'zotero_collection': None,
        'model_loaded': False,
        'model_device': None,  # auto-select
    }.items():
        st.session_state.setdefault(key, default)

    # Main tabs - only show after model is loaded and indexed
    if st.session_state.model_loaded and st.session_state.indexed:
        tab1, tab2 = st.tabs(["Setup", "Search & Highlight"])

        with tab1:
            show_setup_tab()

        with tab2:
            show_search_tab()
    else:
        # Show setup only
        show_setup_tab()

def show_setup_tab():
    """Setup tab for collection selection, model loading, and indexing."""

    st.header("Setup Configuration")

    # Services configuration
    st.subheader("1⃣ Service Endpoints")
    st.text("Configure the local services used for PDF parsing and vector search.")


    col_required_services, col_optional_services = st.columns(2, gap="large")

    with col_required_services:
        st.markdown("##### Required Services")

        grobid_url = st.text_input(
            "GROBID Service URL",
            value=os.environ.get("GROBID_URL", "http://localhost:8070"),
            key="grobid_url_input",
            help="URL of GROBID service. Start with: docker run -p 8070:8070 grobid/grobid"
        )

        qdrant_url = st.text_input(
            "Qdrant Service URL",
            value=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            key="qdrant_url_input",
            help="URL of Qdrant service. Start with: docker run -p 6333:6333 qdrant/qdrant"
        )

    with col_optional_services:
        st.markdown("##### Optional Contextualization")
        use_chunk_contextualization = st.checkbox(
            "Enable chunk contextualization before upsert",
            value=st.session_state.get("use_chunk_contextualization", False),
            key="use_chunk_contextualization_chk",
            help="When enabled, each chunk is enriched with document-level context before vectorization."
        )

        st.session_state.use_chunk_contextualization = use_chunk_contextualization

        st.info(
            "Chunk contextualization enriches each chunk with global document context. "
            "It is unmeasured, slow on long PDFs, and indexing fails if Ollama does not answer."
        )

        ollama_url = st.text_input(
            "Ollama Service URL",
            value=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            key="ollama_url_input",
            disabled=not use_chunk_contextualization,
            help="URL of Ollama service. Start with: docker run -d -v ollama:/root/.ollama -p 11434:11434 ollama/ollama"
        )

    st.markdown("---")

    # Model Selection
    st.subheader("2⃣ Load Models")

    st.markdown(
        """
        <style>
        .st-key-load_model_btn button {
            min-height: 108px;
            font-size: 1.05rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    model_col_left, model_col_right = st.columns([4, 1], gap="large")

    with model_col_left:
        # No model choice: the settings are calibrated on the pipeline's own models.
        st.text("The models are fixed; choose where they run.")

        col_device, col_encode_batch, col_rerank_batch = st.columns(3)

        with col_device:
            device_options = ["auto", "cpu", "cuda", "mps"]
            current_device = "auto" if st.session_state.model_device is None else st.session_state.model_device
            if current_device not in device_options:
                current_device = "auto"

            device_choice = st.selectbox(
                "Device",
                options=device_options,
                index=device_options.index(current_device),
                help="Select compute device. 'auto' picks CUDA, then MPS, then CPU."
            )

        with col_encode_batch:
            encode_batch_size = st.number_input(
                "Encoding batch size",
                min_value=1, max_value=256,
                value=st.session_state.get('encode_batch_size', DEFAULT_ENCODE_BATCH_SIZE),
                help="Chunks encoded per forward pass"
            )

        with col_rerank_batch:
            rerank_batch_size = st.number_input(
                "Reranking batch size",
                min_value=1, max_value=256,
                value=st.session_state.get('rerank_batch_size', DEFAULT_RERANK_BATCH_SIZE),
                help="Query/passage pairs scored per forward pass"
            )

    with model_col_right:
        st.write("")
        st.write("")
        load_model_clicked = st.button(
            "Load Model",
            type="primary",
            width="stretch",
            key="load_model_btn"
        )

    if load_model_clicked:
        with st.spinner("Loading models..."):
            try:
                st.session_state.model_device = None if device_choice == "auto" else device_choice
                st.session_state.encode_batch_size = encode_batch_size
                st.session_state.rerank_batch_size = rerank_batch_size

                st.session_state.rag = ZoteroRAG(
                    grobid_url=grobid_url,
                    qdrant_url=qdrant_url,
                    ollama_url=ollama_url,
                    output_base_dir=st.session_state.output_dir,
                    model_device=st.session_state.model_device,
                    encode_batch_size=encode_batch_size,
                    rerank_batch_size=rerank_batch_size,
                    use_chunk_contextualization=st.session_state.use_chunk_contextualization
                )

                st.session_state.model_loaded = True
                st.session_state.indexed = False  # Reset indexed status

                st.success(
                    "Models loaded\n\n"
                    f"**Batch sizes:** Encoding: {encode_batch_size} | Reranking: {rerank_batch_size}"
                )
                st.rerun()
            except Exception as e:  # noqa: BLE001 - UI boundary: report, never crash the app
                st.error(f"Error loading models: {e}")

    st.markdown("---")

    # Indexing Section - only show if model is loaded
    if st.session_state.model_loaded:
        st.header("Indexed PDFs Manager")
        col_left, col_right = st.columns([3, 1], gap="large")

        with col_left:
            st.markdown("### Indexed PDFs")

            try:
                indexed_titles = st.session_state.rag.get_indexed_pdfs() if st.session_state.rag else []
            except Exception as e:  # noqa: BLE001 - UI boundary: report, never crash the app
                if isinstance(e, ConnectionError) and "Qdrant" in str(e):
                    st.error("Qdrant is not running. Start the service to view indexed PDFs.")
                else:
                    st.error(f"Error fetching indexed PDFs: {e}")
                indexed_titles = []

            new_indexed = len(indexed_titles) > 0
            if st.session_state.indexed != new_indexed:
                st.session_state.indexed = new_indexed
                st.rerun()

            total_pdfs = len(indexed_titles)

            if not st.session_state.rag.consistency_check(indexed_titles):
                st.warning("Consistency check failed: some PDFs in the index may be missing from the cache or have changed. Consider re-indexing to ensure data integrity.")

            filter_text = st.text_input(
                "Search indexed PDFs",
                value="",
                placeholder="Type part of the PDF name...",
                help="Filter list by filename. Useful when index contains many PDFs.",
                key="indexed_pdf_filter"
            )

            def get_pdf_title(item):
                if isinstance(item, dict):
                    return item.get("title", "")
                return str(item)

            if filter_text:
                filtered_pdfs = [
                    item for item in indexed_titles
                    if filter_text.lower() in get_pdf_title(item).lower()
                ]
            else:
                filtered_pdfs = indexed_titles

            if filtered_pdfs:
                total_filtered = len(filtered_pdfs)
                page_size = 10
                items_per_page = page_size * 2
                max_page = max((total_filtered - 1) // items_per_page + 1, 1)

                col_count, col_page = st.columns(2)
                with col_count:
                    st.metric("Total indexed", total_pdfs)
                with col_page:
                    current_page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=max_page,
                        value=1,
                        step=1,
                        key="indexed_pdf_page"
                    )

                start = (current_page - 1) * items_per_page
                end = start + items_per_page
                page_items = filtered_pdfs[start:end]

                left_column_items = page_items[::2]
                right_column_items = page_items[1::2]
                max_rows = max(len(left_column_items), len(right_column_items))

                two_col_rows = []
                for row_idx in range(max_rows):
                    left_value = left_column_items[row_idx] if row_idx < len(left_column_items) else None
                    right_value = right_column_items[row_idx] if row_idx < len(right_column_items) else None
                    two_col_rows.append({
                        "PDF Name (1)": get_pdf_title(left_value) if left_value else "",
                        "PDF Name (2)": get_pdf_title(right_value) if right_value else "",
                    })

                st.caption(f"Showing {start + 1}-{min(end, total_filtered)} of {total_filtered} PDFs")
                st.dataframe(
                    two_col_rows,
                    width="stretch",
                    height=420,
                    column_config={
                        "PDF Name (1)": st.column_config.TextColumn(width="large"),
                        "PDF Name (2)": st.column_config.TextColumn(width="large"),
                    },
                    hide_index=True
                )
            else:
                if total_pdfs == 0:
                    st.info("No indexed PDFs yet. Build your index by adding PDFs from the action panel.")
                else:
                    st.warning("No PDF matches your search filter.")

        with col_right:
            st.markdown("### Actions")

            with st.expander("Add PDFs", expanded=True):
                uploaded_pdfs = st.file_uploader(
                    "Select PDF files to index",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key="add_pdfs_uploader"
                )

                if uploaded_pdfs:
                    st.caption(f"Selected {len(uploaded_pdfs)} PDF file(s).")

                if st.button("Index selected PDFs", width="stretch", key="btn_add_selected_pdfs"):
                    if st.session_state.rag is None:
                        st.error("Load a model first.")
                    elif not uploaded_pdfs:
                        st.warning("Select at least one PDF file.")
                    else:
                        result = st.session_state.rag.ingest_pdfs_from_upload(
                            uploaded_pdfs
                        )
                        run_ingest_and_index(result)

            with st.expander("Add PDFs from Zotero", expanded=False):
                load_zotero_collections()
                if st.session_state.zotero_collections_loaded and st.session_state.zotero_collections:
                    zotero_collection_options = ["All Library"]
                    for coll in st.session_state.zotero_collections:
                        name = coll['name']
                        if coll['parent_id']:
                            parent_name = next((c['name'] for c in st.session_state.zotero_collections
                                            if c['id'] == coll['parent_id']), "Unknown")
                            name = f"{parent_name} > {name}"
                        zotero_collection_options.append(name)

                    selected_zotero_collection = st.selectbox(
                        "Choose which Zotero collection to search",
                        zotero_collection_options,
                        key="zotero_collection_selector"
                    )

                    st.session_state.zotero_collection = None if selected_zotero_collection == "All Library" else selected_zotero_collection.split(" > ")[-1].strip()
                else:
                    st.session_state.zotero_collection = None

                if st.button("Index PDFs from selected collection", width="stretch", key="btn_index_zotero_collection"):
                    if st.session_state.rag is None:
                        st.error("Load a model first.")
                    else:
                        result = st.session_state.rag.ingest_pdfs_from_zotero(
                                st.session_state.zotero_collection
                            )
                        run_ingest_and_index(result)

            with st.expander("Remove one PDF", expanded=False):
                pdf_name_to_delete = st.text_input(
                    "PDF name to delete, must be in the given PDF source",
                    placeholder="example_paper.pdf",
                    key="pdf_name_delete"
                )
                if st.button("Delete PDF", width="stretch", key="btn_delete_pdf"):
                    if not pdf_name_to_delete.strip():
                        st.warning("Insert the exact PDF name.")
                    elif st.session_state.rag is None:
                        st.error("Load a model first.")
                    else:
                        deleted = st.session_state.rag.delete_pdf_by_title(pdf_name_to_delete.strip())
                        if deleted:
                            st.success(f"Deleted PDF '{pdf_name_to_delete}' from index.")
                            time.sleep(2.0)
                            st.rerun()
                        else:
                            st.warning(f"No PDF named '{pdf_name_to_delete}' found in index.")

            with st.expander("Clear full index", expanded=False):
                st.warning("This operation removes all indexed PDFs and vectors.")
                if st.button("Clear all PDFs", width="stretch", key="btn_clear_all_pdfs"):
                    if st.session_state.rag is None:
                        st.error("Load a model first.")
                    else:
                        st.session_state.rag.clear_index()
                        st.session_state.indexed = False
                        st.session_state.model_loaded = False
                        st.session_state.rag = None
                        st.success("Full index clear requested.")
                        st.rerun()

    st.markdown("---")

    # Reset button
    if st.button("Start Over", width="stretch"):
        zotero_collections = st.session_state.get('zotero_collections', [])
        zotero_collections_loaded = st.session_state.get('zotero_collections_loaded', False)
        st.session_state.clear()
        st.session_state.zotero_collections_loaded = zotero_collections_loaded
        st.session_state.zotero_collections = zotero_collections
        st.rerun()

def format_time(seconds: float) -> str:
    """Format a stage duration: seconds, or minutes and seconds."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


def stage_progress(label):
    """A status line and a bar for one stage, and the (current, total, message)
    callback that drives them with elapsed and remaining time."""
    status, bar = st.empty(), st.progress(0)
    started = None

    def update(current, total, message):
        nonlocal started
        if started is None:
            started = time.time()
        progress = current / total if total > 0 else 0.0
        bar.progress(progress)
        elapsed = time.time() - started
        if progress >= 1.0:
            time_info = f"Completed in {format_time(elapsed)}"
        elif current > 0:
            time_info = (f"Elapsed: {format_time(elapsed)} | "
                         f"Remaining: ~{format_time(elapsed / progress - elapsed)}")
        else:
            time_info = ""
        status.text(f"{label}: {message} {time_info}")

    return status, bar, update


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    """``#rrggbb`` as an RGB tuple in 0-1, the form PyMuPDF takes."""
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def show_search_tab():
    """Search and highlight tab."""

    st.header("Search Your Library")

    # No question type and no parameter widgets: the pipeline's settings are
    # fixed, measured values (settings.py), not something to tune per search.
    query = st.text_input(
        "Enter your question",
        placeholder="What are the main findings about...",
        key="search_input"
    )

    st.markdown("---")

    col_search, col_clear = st.columns([1, 4])
    with col_search:
        search_clicked = st.button("Search", type="primary", width="stretch")
    with col_clear:
        if st.button("Clear Results", width="stretch"):
            st.session_state.search_results = []
            st.session_state.current_answer = 0
            st.rerun()

    if search_clicked and query:
        st.markdown("#### Processing Pipeline")
        rerank_status, rerank_progress_bar, rerank_callback = stage_progress("Stage 1 - Reranking")
        qa_status, qa_progress_bar, qa_callback = stage_progress("Stage 2 - QA Extraction")

        try:
            st.session_state.search_results = st.session_state.rag.answer_question(
                question=query,
                progress_callback=qa_callback,
                rerank_callback=rerank_callback,
            )

            st.session_state.search_run_id = st.session_state.get('search_run_id', 0) + 1
            st.session_state.highlight_selected = {
                i: True for i in range(len(st.session_state.search_results))
            }

            # Mark completion
            rerank_progress_bar.progress(1.0)
            qa_progress_bar.progress(1.0)
            rerank_status.text("Stage 1 - Reranking: Complete!")
            qa_status.text("Stage 2 - QA Extraction: Complete!")

            st.session_state.current_answer = 0
            st.session_state.current_query = query
        except Exception as e:  # noqa: BLE001 - UI boundary: report, never crash the app
            st.error(f"Search failed: {e}")
            st.session_state.search_results = []
            with st.expander("Show full error"):
                st.exception(e)
        finally:
            time.sleep(1.0)
            rerank_status.empty()
            rerank_progress_bar.empty()
            qa_status.empty()
            qa_progress_bar.empty()

    # Display results
    if st.session_state.search_results:
        if 'search_run_id' not in st.session_state:
            st.session_state.search_run_id = 1
        if ('highlight_selected' not in st.session_state or
            len(st.session_state.highlight_selected) != len(st.session_state.search_results)):
            st.session_state.highlight_selected = {
                i: True for i in range(len(st.session_state.search_results))
            }

        st.success(f"Found {len(st.session_state.search_results)} answers for: *{st.session_state.current_query}*")

        # Navigation and actions
        st.write("")
        st.write("")
        st.write("")
        col1, col2, col3, spaceCol, col4 = st.columns([1, 1, 1, 3, 1], gap="large")

        with col1:
            if st.button("Previous"):
                # Wrap around: if at first result, go to last
                st.session_state.current_answer = (st.session_state.current_answer - 1) % len(st.session_state.search_results)
                st.rerun()

        with col2:
            st.markdown(f"**Answer {st.session_state.current_answer + 1} / {len(st.session_state.search_results)}**")

        with col3:
            if st.button("Next "):
                # Wrap around: if at last result, go to first
                st.session_state.current_answer = (st.session_state.current_answer + 1) % len(st.session_state.search_results)
                st.rerun()

        with spaceCol:
            pass # Empty column for spacing

        with col4:
            if st.button("Open PDF"):
                answer = st.session_state.search_results[st.session_state.current_answer]
                pdf_path = answer.pdf_path
                try:
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', pdf_path], check=False)
                    elif sys.platform == 'win32':  # Windows
                        os.startfile(pdf_path)
                    else:  # Linux
                        subprocess.run(['xdg-open', pdf_path], check=False)
                    st.success(f"Opened PDF at page {answer.page_number + 1}")
                except OSError as e:
                    st.error(f"Could not open PDF: {e}")

        # Highlight color selection for Highlight Selected
        st.write("")
        st.write("")
        st.subheader("Highlight Color")

        col_color, col_highlight = st.columns([3, 1], gap="large")

        with col_color:
            highlight_color = hex_to_rgb(st.color_picker(
                "Highlight color", value="#ffff00", key="highlight_color"))

        with col_highlight:
            if st.button("Highlight Selected Articles", width="stretch"):
                selected_answers = [
                    ans for i, ans in enumerate(st.session_state.search_results)
                    if st.session_state.highlight_selected.get(i)
                ]

                if not selected_answers:
                    st.warning("Select at least one answer to highlight.")
                else:
                    output_dir = os.path.join(st.session_state.output_dir, "highlighted_results")
                    os.makedirs(output_dir, exist_ok=True)

                    pdfs_answers = defaultdict(list)
                    for answer in selected_answers:
                        answer.color = highlight_color
                        pdfs_answers[answer.pdf_path].append(answer)

                    highlighted_paths = []
                    failed_highlights = []
                    progress_bar = st.progress(0)

                    for idx, (pdf_path, answers) in enumerate(pdfs_answers.items()):
                        display_title = answers[0].title or os.path.splitext(os.path.basename(pdf_path))[0]
                        safe_title = sanitize_filename(display_title)
                        output_filename = f"{safe_title}_highlighted.pdf"
                        output_path = os.path.join(output_dir, output_filename)

                        result_path = highlight_pdf(answers, output_path)
                        if result_path:
                            highlighted_paths.append(result_path)
                        else:
                            failed_highlights.append(display_title)

                        progress_bar.progress((idx + 1) / len(pdfs_answers))

                    progress_bar.empty()

                    if failed_highlights:
                        st.error(f"Failed to highlight {len(failed_highlights)} PDF(s)")
                        with st.expander("Show failed PDFs"):
                            for pdf_name in failed_highlights:
                                st.text(f"{pdf_name}")

                    if highlighted_paths:
                        st.success(f"Successfully highlighted {len(highlighted_paths)} PDF(s)")
                        with st.expander("Show highlighted files"):
                            for path in highlighted_paths:
                                st.text(f"{os.path.basename(path)}")
                            st.text(f"Location: {output_dir}")

        # Current result display
        st.markdown("---")
        answer = st.session_state.search_results[st.session_state.current_answer]

        st.subheader(f"{answer.title}")
        info_col, select_col = st.columns([5, 1], gap="large")

        with info_col:
            st.markdown(
                f"""**PDF**: {answer.title}<br>
                **Page**: {answer.page_number + 1} |
                **Section**: {answer.section or 'Unknown'} |
                **Retrieval Score**: {answer.retrieval_score:.4f} |
                **Rerank Score**: {answer.rerank_score:.4f} |
                **QA Score**: {answer.score:.4f}""",
                unsafe_allow_html=True
            )

        with select_col:
            selection_key = f"highlight_select_{st.session_state.search_run_id}_{st.session_state.current_answer}"
            highlight_this = st.checkbox(
                "Highlight this answer",
                value=st.session_state.highlight_selected.get(st.session_state.current_answer, True),
                key=selection_key
            )
            st.session_state.highlight_selected[st.session_state.current_answer] = highlight_this

        # Answer display
        st.subheader("Answer")
        st.info(answer.text)

        # Context
        st.subheader("Context (Full Chunk)")
        st.text_area(
            "Full chunk containing the answer",
            value=answer.context,
            height=250,
            disabled=True,
            label_visibility="collapsed",
            key=f"context_{st.session_state.current_answer}"
        )

    elif query and search_clicked:
        st.warning("No results found")
        st.info("No passage in the library is relevant enough to answer this. "
                'Try a specific question (e.g., "What is X?" instead of "Tell me about X").')
    elif not query and search_clicked:
        st.info("Enter a question above and click Search to get started")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Zotero RAG Navigator • Built with Streamlit</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()