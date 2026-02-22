from unittest.mock import MagicMock, patch

import numpy as np
from sqlalchemy.orm import Session

import pipeline


def test_process_embeds_and_stores_each_expedition():
    mock_session = MagicMock(spec=Session)

    with (
        patch("pipeline.db.load_expedition") as mock_load,
        patch("pipeline.db.upsert_embedding") as mock_upsert,
        patch("pipeline.embedder.compute") as mock_compute,
    ):
        mock_load.return_value = {"id": 1, "peak_id": 1, "members": []}
        mock_compute.return_value = np.zeros(512, dtype=np.float32)

        pipeline.process(mock_session, [1])

        mock_load.assert_called_once_with(mock_session, 1)
        mock_compute.assert_called_once_with({"id": 1, "peak_id": 1, "members": []})
        mock_upsert.assert_called_once()
        call_args = mock_upsert.call_args
        assert call_args.args[1] == 1         # expedition_id
        assert call_args.args[3] == "stub-v0" # model_version


def test_process_commits_once_after_batch():
    mock_session = MagicMock(spec=Session)

    with (
        patch("pipeline.db.load_expedition") as mock_load,
        patch("pipeline.db.upsert_embedding"),
        patch("pipeline.embedder.compute") as mock_compute,
    ):
        mock_load.side_effect = [
            {"id": 1, "members": []},
            {"id": 2, "members": []},
            {"id": 3, "members": []},
        ]
        mock_compute.return_value = np.zeros(512, dtype=np.float32)

        pipeline.process(mock_session, [1, 2, 3])

        mock_session.commit.assert_called_once()


def test_process_skips_missing_expedition():
    mock_session = MagicMock(spec=Session)

    with (
        patch("pipeline.db.load_expedition") as mock_load,
        patch("pipeline.db.upsert_embedding") as mock_upsert,
    ):
        mock_load.return_value = None

        pipeline.process(mock_session, [999])

        mock_upsert.assert_not_called()
        mock_session.commit.assert_called_once()


def test_process_handles_empty_list():
    mock_session = MagicMock(spec=Session)

    pipeline.process(mock_session, [])

    mock_session.commit.assert_called_once()
