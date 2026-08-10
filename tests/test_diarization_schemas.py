import pytest

from src.diarization.schemas import DiarizationResult, Speaker, SpeakerTurn


def test_valid_single_speaker_result() -> None:
    result = DiarizationResult(
        schema_version="1.0",
        audio_id="audio-1",
        speakers=[Speaker(speaker_id="speaker-1")],
        turns=[
            SpeakerTurn(
                speaker_id="speaker-1",
                start_s=0.0,
                end_s=2.5,
                confidence=0.95,
            )
        ],
    )

    assert result.audio_id == "audio-1"
    assert result.speakers == [Speaker(speaker_id="speaker-1")]
    assert result.turns[0].end_s == 2.5


def test_valid_multiple_speaker_result() -> None:
    result = DiarizationResult(
        schema_version="1.0",
        audio_id="audio-2",
        speakers=[Speaker("speaker-1"), Speaker("speaker-2")],
        turns=[
            SpeakerTurn("speaker-1", 0.0, 1.0),
            SpeakerTurn("speaker-2", 1.0, 2.0),
        ],
    )

    assert [speaker.speaker_id for speaker in result.speakers] == [
        "speaker-1",
        "speaker-2",
    ]


def test_overlapping_turns_are_accepted() -> None:
    result = DiarizationResult(
        schema_version="1.0",
        audio_id="audio-overlap",
        speakers=[Speaker("speaker-1"), Speaker("speaker-2")],
        turns=[
            SpeakerTurn("speaker-1", 0.0, 2.0),
            SpeakerTurn("speaker-2", 1.0, 3.0),
        ],
    )

    assert len(result.turns) == 2


def test_negative_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SpeakerTurn("speaker-1", -0.1, 1.0)


@pytest.mark.parametrize(
    ("start_s", "end_s"),
    [(1.0, 1.0), (2.0, 1.0)],
)
def test_end_at_or_before_start_is_rejected(start_s: float, end_s: float) -> None:
    with pytest.raises(ValueError, match="greater than start_s"):
        SpeakerTurn("speaker-1", start_s, end_s)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_outside_zero_to_one_is_rejected(confidence: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        SpeakerTurn("speaker-1", 0.0, 1.0, confidence=confidence)


def test_turn_referencing_undeclared_speaker_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown speaker_id"):
        DiarizationResult(
            schema_version="1.0",
            audio_id="audio-unknown-speaker",
            speakers=[Speaker("speaker-1")],
            turns=[SpeakerTurn("speaker-2", 0.0, 1.0)],
        )


def test_duplicate_speaker_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate speaker_id"):
        DiarizationResult(
            schema_version="1.0",
            audio_id="audio-duplicate-speaker",
            speakers=[Speaker("speaker-1"), Speaker("speaker-1")],
            turns=[],
        )
