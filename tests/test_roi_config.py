from xueliu_ai.capture.roi_config import (
    Roi,
    ScreenProfile,
    clear_rois,
    load_screen_profile,
    save_screen_profile,
)


def test_clear_rois_preserves_table_and_clears_selected_regions(tmp_path) -> None:
    path = tmp_path / "screen_profile.yaml"
    save_screen_profile(
        ScreenProfile(
            monitor=1,
            rois={
                "table": Roi(10, 20, 1000, 800),
                "my_hand": Roi(100, 700, 600, 80),
                "discards": Roi(250, 200, 500, 400),
            },
        ),
        path,
    )

    clear_rois(("my_hand", "discards"), path)
    profile = load_screen_profile(path)

    assert profile.rois["table"] == Roi(10, 20, 1000, 800)
    assert profile.rois["my_hand"].is_empty
    assert profile.rois["discards"].is_empty
