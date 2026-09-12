"""chunk_text: the character-window chunker used during ingest."""
from secondbrain.ingest import chunk_text


def test_empty_or_whitespace_yields_no_chunks():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   \n\t  ", 100, 10) == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world", 100, 10) == ["hello world"]


def test_long_text_splits_into_bounded_overlapping_chunks():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    # No chunk exceeds the window size.
    assert all(len(c) <= 200 for c in chunks)
    # Whole text remains covered (cheap content check on a distinctive token).
    assert "word" in chunks[0] and "word" in chunks[-1]


def test_prefers_newline_break_near_window_edge():
    # A newline sits inside the break window, so the first chunk should end there.
    text = "alpha beta gamma\n" + "x" * 300
    chunks = chunk_text(text, size=20, overlap=8)
    assert chunks[0] == "alpha beta gamma"


class TestContextualHeaders:
    """A chunk from the middle of a long document arrived with no orientation.

    Body text alone was everything retrieval had to match on and everything the
    model was given. A chunk reading "Stop the launchd job, wait for the port to
    close" does not say which service, and neither the embedding nor the keyword
    index could tell it apart from the same sentence in another runbook. The
    measured consequence is on the hard benchmark: asked which agent owns QA,
    retrieval returns the superseded routing document alongside the current one,
    because nothing in the chunk bodies says which is which.

    The header is a retrieval aid, not source text. It is embedded and indexed;
    it is not what a citation quotes.
    """

    def test_a_chunk_carries_the_document_title_and_its_nearest_heading(self):
        from secondbrain.ingest import chunk_document

        text = "# Gateway runbook\n\n## Restarting\n\nStop the launchd job.\n"
        chunks = chunk_document(text, 1200, 150, name="runbook-gateway.md")

        assert chunks[0].title == "Gateway runbook"
        assert chunks[0].section == "Restarting"
        assert chunks[0].header == "Gateway runbook › Restarting"

    def test_the_stored_body_does_not_contain_the_header(self):
        from secondbrain.ingest import chunk_document

        text = "# Gateway runbook\n\n## Restarting\n\nStop the launchd job.\n"
        chunks = chunk_document(text, 1200, 150, name="runbook-gateway.md")

        assert "›" not in chunks[0].body
        assert chunks[0].body.endswith("Stop the launchd job.")
        assert chunks[0].embedded.startswith("Gateway runbook › Restarting\n\n")
        assert chunks[0].embedded.endswith(chunks[0].body)

    def test_each_chunk_gets_the_heading_it_actually_sits_under(self):
        from secondbrain.ingest import chunk_document

        text = (
            "# Runbook\n\n"
            "## Budgets\n\n" + "budget detail. " * 80 + "\n\n"
            "## Timeouts\n\n" + "timeout detail. " * 80 + "\n"
        )
        chunks = chunk_document(text, 400, 40, name="r.md")

        sections = [chunk.section for chunk in chunks]
        assert "Budgets" in sections and "Timeouts" in sections
        for chunk in chunks:
            if "budget detail" in chunk.body and "timeout detail" not in chunk.body:
                assert chunk.section == "Budgets"
            if "timeout detail" in chunk.body and "budget detail" not in chunk.body:
                assert chunk.section == "Timeouts"

    def test_a_document_with_no_h1_falls_back_to_the_file_stem(self):
        from secondbrain.ingest import chunk_document

        chunks = chunk_document("Just prose, no headings at all.", 1200, 150, name="meeting-notes.md")

        assert chunks[0].title == "meeting-notes"
        assert chunks[0].section is None
        assert chunks[0].header == "meeting-notes"

    def test_bodies_are_exactly_what_the_old_chunker_produced(self):
        """The header is additive. Changing the split would be a different change."""
        from secondbrain.ingest import chunk_document, chunk_text

        text = "# Title\n\n## One\n\n" + ("alpha beta gamma. " * 200) + "\n\n## Two\n\n" + ("delta. " * 200)

        assert [chunk.body for chunk in chunk_document(text, 500, 60, name="t.md")] == chunk_text(text, 500, 60)

    def test_a_setext_or_deep_heading_does_not_become_the_section(self):
        """Only ATX H2/H3 are sections; a '#####' is too fine and '#' is the title."""
        from secondbrain.ingest import chunk_document

        text = "# Doc\n\n##### Footnote\n\nbody text here\n"
        chunks = chunk_document(text, 1200, 150, name="d.md")

        assert chunks[0].section is None

    def test_a_hash_inside_a_fenced_code_block_is_not_a_heading(self):
        """`# comment` in a shell block is a comment, and it was becoming a section."""
        from secondbrain.ingest import chunk_document

        text = "# Doc\n\n## Real section\n\n```bash\n## not a heading\necho hi\n```\n\nprose\n"
        chunks = chunk_document(text, 1200, 150, name="d.md")

        assert chunks[0].section == "Real section"

    def test_pdf_chunks_record_the_page_they_start_on(self):
        from secondbrain.ingest import chunk_document

        # Two pages, the boundary offsets given the way a PDF read reports them.
        text = "page one text\n\npage two text"
        chunks = chunk_document(text, 20, 2, name="doc.pdf", page_starts=[0, len("page one text\n\n")])

        assert chunks[0].page == 1
        assert chunks[-1].page == 2

    def test_a_non_pdf_chunk_has_no_page(self):
        from secondbrain.ingest import chunk_document

        chunks = chunk_document("some prose", 1200, 150, name="notes.md")

        assert chunks[0].page is None
